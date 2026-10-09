// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/memory/tlsf_pool.h"

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>

#include "iree/async/frontier_tracker.h"
#include "iree/async/notification.h"
#include "iree/async/proactor.h"
#include "iree/async/proactor_platform.h"
#include "iree/hal/api.h"
#include "iree/hal/memory/cpu_slab_provider.h"
#include "iree/hal/memory/fixed_block_pool.h"
#include "iree/hal/memory/maintenance_thread.h"
#include "iree/hal/memory/passthrough_pool.h"
#include "iree/hal/memory/slab_cache.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

static iree_hal_memory_maintenance_t* test_maintenance() {
  static iree_hal_memory_maintenance_t* maintenance = nullptr;
  if (!maintenance) {
    IREE_CHECK_OK(iree_hal_memory_maintenance_thread_create(
        {}, iree_allocator_system(), &maintenance));
    atexit([] {
      iree_hal_memory_maintenance_release(maintenance);
      maintenance = nullptr;
    });
  }
  return maintenance;
}

// An ordered observation point for tests of asynchronous native retirement.
// Pool trim itself never waits for work on a shared maintenance executor.
static void WaitForMaintenance() {
  struct Barrier : iree_hal_memory_maintenance_entry_t {
    // Protects completion and the callback's final notification access.
    std::mutex mutex;
    // Wakes the observing test after preceding native work finishes.
    std::condition_variable condition;
    // Set by the maintenance callback under mutex.
    bool complete = false;
  } barrier;
  barrier.fn = [](iree_hal_memory_maintenance_entry_t* entry) {
    auto* barrier = static_cast<Barrier*>(entry);
    std::lock_guard<std::mutex> lock(barrier->mutex);
    barrier->complete = true;
    barrier->condition.notify_all();
  };
  iree_hal_memory_maintenance_enqueue(test_maintenance(), &barrier);
  std::unique_lock<std::mutex> lock(barrier.mutex);
  barrier.condition.wait(lock, [&] { return barrier.complete; });
}

static iree_status_t CreateNativePool(iree_hal_slab_provider_t* provider,
                                      iree_async_notification_t* notification,
                                      iree_async_frontier_tracker_t* tracker,
                                      iree_hal_pool_epoch_query_t epoch_query,
                                      iree_hal_pool_t** out_pool) {
  iree_hal_passthrough_pool_options_t options = {.epoch_query = epoch_query};
  return iree_hal_passthrough_pool_create(options, provider, notification,
                                          tracker, test_maintenance(),
                                          iree_allocator_system(), out_pool);
}

// Uses the production native leaf. Only TLSF metadata uses the supplied host
// allocator, allowing metadata-allocation tests to observe that policy alone.
static iree_status_t CreateTLSFPool(iree_hal_tlsf_pool_options_t options,
                                    iree_hal_slab_provider_t* provider,
                                    iree_async_notification_t* notification,
                                    iree_async_frontier_tracker_t* tracker,
                                    iree_hal_pool_epoch_query_t epoch_query,
                                    iree_allocator_t allocator,
                                    iree_hal_pool_t** out_pool) {
  iree_hal_pool_t* backing_pool = nullptr;
  IREE_RETURN_IF_ERROR(CreateNativePool(provider, notification, tracker,
                                        epoch_query, &backing_pool));
  iree_status_t status =
      iree_hal_tlsf_pool_create(backing_pool, &options, allocator, out_pool);
  iree_hal_pool_release(backing_pool);
  return status;
}

// A finite arena retains its native source until all child operations finish.
// The caller releases the child before releasing the returned source pool.
static iree_status_t CreateFiniteTLSFPool(
    iree_hal_tlsf_pool_options_t options, iree_hal_slab_provider_t* provider,
    iree_async_notification_t* notification,
    iree_async_frontier_tracker_t* tracker, iree_allocator_t allocator,
    iree_hal_pool_t** out_backing_pool, iree_hal_pool_t** out_pool) {
  IREE_RETURN_IF_ERROR(
      CreateNativePool(provider, notification, tracker, {}, out_backing_pool));
  iree_hal_pool_reservation_request_t request;
  IREE_RETURN_IF_ERROR(iree_hal_tlsf_pool_query_backing_request(
      *out_backing_pool, &options, &request));
  iree_hal_buffer_t* buffer = nullptr;
  IREE_RETURN_IF_ERROR(iree_hal_pool_allocate_buffer(
      *out_backing_pool, request.params, request.allocation_size,
      iree_infinite_timeout(), &buffer));
  options.tlsf_options.range_length = 0;
  iree_status_t status = iree_hal_tlsf_pool_create_from_buffer(
      buffer, 0, IREE_HAL_WHOLE_BUFFER, &options, allocator, out_pool);
  iree_hal_buffer_release(buffer);
  return status;
}

static iree_async_frontier_tracker_t* test_frontier_tracker() {
  static iree_async_frontier_tracker_t* tracker = nullptr;
  if (!tracker) {
    IREE_CHECK_OK(iree_async_frontier_tracker_create(
        iree_async_frontier_tracker_options_default(), iree_allocator_system(),
        &tracker));
    for (uint8_t queue_index = 0; queue_index < 2; ++queue_index) {
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

typedef struct iree_hal_test_epoch_query_t {
  // Only axis recognized by the test query.
  iree_async_axis_t axis;
  // Greatest epoch reported complete on |axis|.
  uint64_t completed_epoch;
  // Number of completion queries received.
  iree_host_size_t query_count;
} iree_hal_test_epoch_query_t;

static bool iree_hal_test_epoch_query(void* user_data, iree_async_axis_t axis,
                                      uint64_t epoch) {
  iree_hal_test_epoch_query_t* query = (iree_hal_test_epoch_query_t*)user_data;
  ++query->query_count;
  return axis == query->axis && epoch <= query->completed_epoch;
}

typedef struct iree_hal_test_counting_allocator_t {
  // Allocator that performs the actual memory operations.
  iree_allocator_t backing_allocator;
  // Number of allocation-like commands forwarded to the backing allocator.
  std::atomic<iree_host_size_t> allocation_call_count{0};
  // Number of free commands forwarded to the backing allocator.
  std::atomic<iree_host_size_t> free_call_count{0};
} iree_hal_test_counting_allocator_t;

static iree_status_t iree_hal_test_counting_allocator_ctl(
    void* self, iree_allocator_command_t command, const void* params,
    void** inout_ptr) {
  iree_hal_test_counting_allocator_t* allocator =
      (iree_hal_test_counting_allocator_t*)self;
  switch (command) {
    case IREE_ALLOCATOR_COMMAND_MALLOC:
    case IREE_ALLOCATOR_COMMAND_CALLOC:
    case IREE_ALLOCATOR_COMMAND_REALLOC:
      ++allocator->allocation_call_count;
      break;
    case IREE_ALLOCATOR_COMMAND_FREE:
      ++allocator->free_call_count;
      break;
  }
  return allocator->backing_allocator.ctl(allocator->backing_allocator.self,
                                          command, params, inout_ptr);
}

static iree_allocator_t iree_hal_test_counting_allocator(
    iree_hal_test_counting_allocator_t* allocator) {
  return iree_allocator_t{
      .self = allocator,
      .ctl = iree_hal_test_counting_allocator_ctl,
  };
}

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

  // Whether native buffer view creation fails during materialization.
  bool fail_wrap;

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
  if (provider->fail_wrap) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "native buffer view allocation failed");
  }
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
  out_properties->maintenance_alignment = 1;
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

// Compare native backing identity, independent of allocator directory layout.
static const iree_hal_buffer_backing_facts_t* ReservationBacking(
    iree_hal_pool_t* pool, const iree_hal_pool_reservation_t& reservation) {
  const auto request = MakeReservationRequest(reservation.byte_length, 16);
  iree_hal_buffer_t* buffer = nullptr;
  IREE_CHECK_OK(iree_hal_pool_materialize_reservations(
      pool, 1, &request, &reservation, IREE_HAL_POOL_MATERIALIZE_FLAG_NONE,
      &buffer));
  const auto* backing = iree_hal_buffer_memory_view(buffer).backing;
  iree_hal_buffer_release(buffer);
  return backing;
}

static iree_hal_tlsf_pool_options_t DefaultOptions() {
  iree_hal_tlsf_pool_options_t options = {};
  options.tlsf_options.range_length = 4096;
  options.tlsf_options.alignment = 16;
  options.tlsf_options.initial_block_capacity = 16;
  options.tlsf_options.frontier_capacity = 2;
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

class TLSFPoolTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_hal_cpu_slab_provider_create(
        /*min_alignment=*/0, allocator_, &slab_provider_));
    IREE_ASSERT_OK(iree_async_notification_create(
        test_proactor(), IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification_));
    IREE_ASSERT_OK(CreatePool());
  }

  void TearDown() override {
    iree_hal_pool_release(pool_);
    iree_hal_pool_release(backing_pool_);
    iree_async_notification_release(notification_);
    iree_hal_slab_provider_release(slab_provider_);
  }

  virtual iree_status_t CreatePool() {
    return CreateTLSFPool(DefaultOptions(), slab_provider_, notification_,
                          test_frontier_tracker(), {}, allocator_, &pool_);
  }

  iree_allocator_t allocator_ = iree_allocator_system();
  iree_hal_slab_provider_t* slab_provider_ = nullptr;
  iree_async_notification_t* notification_ = nullptr;
  iree_hal_pool_t* pool_ = nullptr;
  // Native source retained only by finite-arena tests.
  iree_hal_pool_t* backing_pool_ = nullptr;
};

class FiniteTLSFPoolTest : public TLSFPoolTest {
 protected:
  iree_status_t CreatePool() override {
    return CreateFiniteTLSFPool(DefaultOptions(), slab_provider_, notification_,
                                test_frontier_tracker(), allocator_,
                                &backing_pool_, &pool_);
  }
};

TEST_F(TLSFPoolTest, ReserveReleaseFresh) {
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

TEST(TLSFPoolAlignmentTest, CreationRequiresAbsoluteNativeAlignment) {
  iree_hal_test_counting_allocator_t allocations = {iree_allocator_system()};
  iree_allocator_t allocator = iree_hal_test_counting_allocator(&allocations);
  iree_hal_slab_provider_t* provider = nullptr;
  IREE_ASSERT_OK(
      iree_hal_test_opaque_slab_provider_create(allocator, &provider));
  iree_async_notification_t* notification = nullptr;
  IREE_ASSERT_OK(iree_async_notification_create(
      test_proactor(), IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification));

  auto options = DefaultOptions();
  options.tlsf_options.alignment = 2 * IREE_HAL_HEAP_BUFFER_ALIGNMENT;
  iree_hal_pool_t* pool = nullptr;
  const auto allocation_count = allocations.allocation_call_count.load();
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      CreateTLSFPool(options, provider, notification, test_frontier_tracker(),
                     iree_hal_pool_epoch_query_null(), allocator, &pool));
  EXPECT_EQ(pool, nullptr);
  EXPECT_EQ(allocations.allocation_call_count, allocation_count);

  options.tlsf_options.alignment = IREE_HAL_HEAP_BUFFER_ALIGNMENT;
  IREE_ASSERT_OK(
      CreateTLSFPool(options, provider, notification, test_frontier_tracker(),
                     iree_hal_pool_epoch_query_null(), allocator, &pool));
  auto request = MakeReservationRequest(128, IREE_HAL_HEAP_BUFFER_ALIGNMENT);
  request.params.usage |= IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool, request.params, request.allocation_size, iree_infinite_timeout(),
      &buffer));
  iree_hal_buffer_mapping_t mapping = {};
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      buffer, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_ALL,
      IREE_HAL_BUFFER_MAP_FLAG_NONE, 0, 128, &mapping));
  EXPECT_EQ(reinterpret_cast<uintptr_t>(mapping.contents.data) %
                IREE_HAL_HEAP_BUFFER_ALIGNMENT,
            0u);
  memset(mapping.contents.data, 0x6B, 128);
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));
  uint8_t actual[128] = {};
  IREE_ASSERT_OK(iree_hal_buffer_map_read(buffer, 0, actual, sizeof(actual)));
  for (uint8_t value : actual) {
    EXPECT_EQ(value, 0x6B);
  }
  iree_hal_buffer_release(buffer);
  iree_hal_pool_release(pool);
  iree_async_notification_release(notification);
  iree_hal_slab_provider_release(provider);
}

TEST_F(FiniteTLSFPoolTest, ReservationTransactionIsAllOrNoneWithoutGrowth) {
  iree_hal_pool_reservation_t seed_reservation;
  iree_hal_pool_acquire_info_t seed_info;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(AcquireOneReservation(
      pool_, 4096, 16, /*requester_frontier=*/NULL,
      IREE_HAL_POOL_RESERVE_FLAG_NONE, &seed_reservation, &seed_info, &result));
  ReleaseOneReservation(pool_, &seed_reservation, /*death_frontier=*/NULL);

  const iree_hal_pool_reservation_request_t requests[2] = {
      MakeReservationRequest(3072, 16),
      MakeReservationRequest(3072, 16),
  };
  iree_hal_pool_reservation_t failed_reservations[2];
  memset(failed_reservations, 0xA5, sizeof(failed_reservations));
  iree_hal_pool_reservation_t original_reservations[2];
  memcpy(original_reservations, failed_reservations,
         sizeof(original_reservations));
  iree_hal_pool_acquire_info_t failed_infos[2];
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations(
      pool_, IREE_ARRAYSIZE(requests), requests,
      /*requester_frontier=*/NULL, IREE_HAL_POOL_RESERVE_FLAG_DISALLOW_GROWTH,
      failed_reservations, failed_infos, &result));

  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_EXHAUSTED);
  EXPECT_EQ(memcmp(failed_reservations, original_reservations,
                   sizeof(failed_reservations)),
            0);
  EXPECT_EQ(failed_infos[0].result, IREE_HAL_POOL_ACQUIRE_NONE);
  EXPECT_EQ(failed_infos[1].result, IREE_HAL_POOL_ACQUIRE_EXHAUSTED);
  EXPECT_EQ(failed_infos[1].flags, IREE_HAL_POOL_ACQUIRE_FLAG_NONE);

  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool_, &stats);
  EXPECT_EQ(stats.reservation_count, 0u);
  EXPECT_EQ(stats.bytes_reserved, 0u);
  EXPECT_EQ(stats.reserve_count, 1u);
  EXPECT_EQ(stats.release_count, 1u);

  iree_hal_pool_reservation_t reservation;
  iree_hal_pool_acquire_info_t info;
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations(
      pool_, 1, requests, /*requester_frontier=*/NULL,
      IREE_HAL_POOL_RESERVE_FLAG_DISALLOW_GROWTH, &reservation, &info,
      &result));
  EXPECT_NE(reservation.block_handle, 0u);
  iree_hal_pool_release_reservations(pool_, 1, &reservation,
                                     /*death_frontier=*/NULL);
}

TEST_F(TLSFPoolTest, MaterializationTransactionTransfersAllReservations) {
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

TEST_F(TLSFPoolTest, ReleaseWithObserverSignalsNotification) {
  iree_hal_pool_reservation_t reservation;
  iree_hal_pool_acquire_info_t reserve_info;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(AcquireOneReservation(
      pool_, 128, 16, /*requester_frontier=*/NULL,
      IREE_HAL_POOL_RESERVE_FLAG_NONE, &reservation, &reserve_info, &result));

  const uint32_t wait_token =
      iree_async_notification_begin_observe(iree_hal_pool_notification(pool_));
  ReleaseOneReservation(pool_, &reservation, NULL);
  EXPECT_TRUE(iree_async_notification_wait_for_token(
      iree_hal_pool_notification(pool_), wait_token, iree_make_timeout_ms(0)));
  iree_async_notification_end_observe(iree_hal_pool_notification(pool_));
}

TEST_F(FiniteTLSFPoolTest, ReserveRejectsOversizedAllocation) {
  iree_hal_pool_reservation_t reservation;
  iree_hal_pool_acquire_info_t reserve_info;
  iree_hal_pool_acquire_result_t result;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      AcquireOneReservation(pool_, 8192, 16, /*requester_frontier=*/NULL,
                            IREE_HAL_POOL_RESERVE_FLAG_NONE, &reservation,
                            &reserve_info, &result));
}

TEST_F(TLSFPoolTest, ReserveGrowsForLivePressure) {
  iree_hal_pool_reservation_t first_reservation;
  iree_hal_pool_acquire_info_t reserve_info;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(
      AcquireOneReservation(pool_, 4096, 16, /*requester_frontier=*/NULL,
                            IREE_HAL_POOL_RESERVE_FLAG_NONE, &first_reservation,
                            &reserve_info, &result));
  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);

  iree_hal_pool_reservation_t second_reservation;
  IREE_ASSERT_OK(
      AcquireOneReservation(pool_, 4096, 16, /*requester_frontier=*/NULL,
                            IREE_HAL_POOL_RESERVE_FLAG_NONE,
                            &second_reservation, &reserve_info, &result));
  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);
  EXPECT_NE(ReservationBacking(pool_, second_reservation),
            ReservationBacking(pool_, first_reservation));

  ReleaseOneReservation(pool_, &second_reservation, NULL);
  ReleaseOneReservation(pool_, &first_reservation, NULL);
}

TEST_F(TLSFPoolTest, TrimPreservesLiveSlab) {
  iree_hal_pool_reservation_t first_reservation;
  iree_hal_pool_acquire_info_t reserve_info;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(
      AcquireOneReservation(pool_, 4096, 16, /*requester_frontier=*/NULL,
                            IREE_HAL_POOL_RESERVE_FLAG_NONE, &first_reservation,
                            &reserve_info, &result));
  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);

  iree_hal_pool_reservation_t second_reservation;
  IREE_ASSERT_OK(
      AcquireOneReservation(pool_, 16, 16, /*requester_frontier=*/NULL,
                            IREE_HAL_POOL_RESERVE_FLAG_NONE,
                            &second_reservation, &reserve_info, &result));
  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);
  EXPECT_NE(ReservationBacking(pool_, first_reservation),
            ReservationBacking(pool_, second_reservation));

  ReleaseOneReservation(pool_, &first_reservation, NULL);
  iree_hal_pool_trim(pool_, IREE_HAL_POOL_TRIM_FLAG_ALL, 16);

  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool_, &stats);
  EXPECT_EQ(stats.slab_count, 1u);
  EXPECT_EQ(stats.bytes_committed, 4096u);
  EXPECT_EQ(stats.reservation_count, 1u);

  iree_hal_buffer_params_t params = {0};
  params.usage =
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
  params.type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  iree_hal_buffer_t* buffer = NULL;
  IREE_ASSERT_OK(MaterializeOneReservation(
      pool_, params, &second_reservation,
      IREE_HAL_POOL_MATERIALIZE_FLAG_TRANSFER_RESERVATION_OWNERSHIP, &buffer));
  ASSERT_NE(buffer, nullptr);
  iree_hal_buffer_release(buffer);
}

TEST_F(TLSFPoolTest, TrimRetainsByteThresholdForIdleSlabs) {
  iree_hal_pool_t* native = nullptr;
  IREE_ASSERT_OK(CreateNativePool(slab_provider_, notification_,
                                  test_frontier_tracker(), {}, &native));
  iree_hal_slab_cache_options_t cache_options = {
      .slab = MakeReservationRequest(4096, 16), .max_count = 4};
  iree_hal_pool_t* cache = nullptr;
  IREE_ASSERT_OK(
      iree_hal_slab_cache_create(native, &cache_options, allocator_, &cache));
  iree_hal_pool_release(native);
  iree_hal_pool_release(pool_);
  const auto options = DefaultOptions();
  IREE_ASSERT_OK(
      iree_hal_tlsf_pool_create(cache, &options, allocator_, &pool_));
  iree_hal_pool_acquire_info_t reserve_info;
  iree_hal_pool_acquire_result_t result;
  iree_hal_pool_reservation_t first_reservation;
  IREE_ASSERT_OK(
      AcquireOneReservation(pool_, 4096, 16, /*requester_frontier=*/NULL,
                            IREE_HAL_POOL_RESERVE_FLAG_NONE, &first_reservation,
                            &reserve_info, &result));
  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);

  iree_hal_pool_reservation_t second_reservation;
  IREE_ASSERT_OK(
      AcquireOneReservation(pool_, 4096, 16, /*requester_frontier=*/NULL,
                            IREE_HAL_POOL_RESERVE_FLAG_NONE,
                            &second_reservation, &reserve_info, &result));
  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);

  ReleaseOneReservation(pool_, &first_reservation, NULL);
  ReleaseOneReservation(pool_, &second_reservation, NULL);

  WaitForMaintenance();
  iree_hal_pool_stats_t stats;
  iree_hal_pool_trim(cache, IREE_HAL_POOL_TRIM_FLAG_ALL, 16384);
  WaitForMaintenance();
  iree_hal_pool_query_stats(cache, &stats);
  EXPECT_EQ(stats.slab_count, 2u);
  EXPECT_EQ(stats.bytes_committed, 8192u);

  iree_hal_pool_trim(cache, IREE_HAL_POOL_TRIM_FLAG_ALL, 4097);
  WaitForMaintenance();
  iree_hal_pool_query_stats(cache, &stats);
  EXPECT_EQ(stats.slab_count, 2u);
  EXPECT_EQ(stats.bytes_committed, 8192u);

  iree_hal_pool_trim(cache, IREE_HAL_POOL_TRIM_FLAG_ALL, 4096);
  WaitForMaintenance();
  iree_hal_pool_query_stats(cache, &stats);
  EXPECT_EQ(stats.slab_count, 1u);
  EXPECT_EQ(stats.bytes_committed, 4096u);

  iree_hal_pool_trim(cache, IREE_HAL_POOL_TRIM_FLAG_ALL, 0);
  WaitForMaintenance();
  iree_hal_pool_query_stats(cache, &stats);
  EXPECT_EQ(stats.slab_count, 0u);
  EXPECT_EQ(stats.bytes_committed, 0u);

  const auto request = MakeReservationRequest(128, 16);
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool_, request.params, request.allocation_size, iree_immediate_timeout(),
      &buffer));
  iree_hal_pool_query_stats(pool_, &stats);
  EXPECT_EQ(stats.slab_count, 1u);
  EXPECT_EQ(stats.reservation_count, 1u);
  iree_hal_buffer_release(buffer);
  iree_hal_pool_release(cache);
}

TEST(TLSFPool, TrimReturnsPendingHistoryToNativeOwner) {
  iree_async_frontier_tracker_t* tracker = nullptr;
  IREE_ASSERT_OK(iree_async_frontier_tracker_create(
      iree_async_frontier_tracker_options_default(), iree_allocator_system(),
      &tracker));
  IREE_ASSERT_OK(iree_async_frontier_tracker_register_axis(
      tracker, TestQueueAxis(0), nullptr));
  iree_hal_slab_provider_t* provider = nullptr;
  IREE_ASSERT_OK(
      iree_hal_cpu_slab_provider_create(0, iree_allocator_system(), &provider));
  iree_async_notification_t* notification = nullptr;
  IREE_ASSERT_OK(iree_async_notification_create(
      test_proactor(), IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification));
  iree_hal_pool_t* backing_pool = nullptr;
  IREE_ASSERT_OK(
      CreateNativePool(provider, notification, tracker, {}, &backing_pool));
  auto options = DefaultOptions();
  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(iree_hal_tlsf_pool_create(backing_pool, &options,
                                           iree_allocator_system(), &pool));
  iree_hal_pool_reservation_t reservation;
  iree_hal_pool_acquire_info_t info;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(AcquireOneReservation(pool, 4096, 16, nullptr,
                                       IREE_HAL_POOL_RESERVE_FLAG_NONE,
                                       &reservation, &info, &result));
  ASSERT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);
  MAKE_FRONTIER(death, 1, E(TestQueueAxis(0), 42));
  ReleaseOneReservation(pool, &reservation, death);
  iree_hal_pool_trim(pool, IREE_HAL_POOL_TRIM_FLAG_ALL, 0);
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool, &stats);
  EXPECT_EQ(stats.slab_count, 0u);
  EXPECT_EQ(stats.bytes_committed, 0u);
  iree_hal_pool_release(pool);

  // The child can disappear after returning its exact history. Its native
  // owner still holds the backing until that prerequisite actually resolves.
  iree_hal_pool_query_stats(backing_pool, &stats);
  EXPECT_EQ(stats.reservation_count, 0u);
  EXPECT_EQ(stats.bytes_committed, 4096u);
  iree_async_frontier_tracker_advance(tracker, TestQueueAxis(0), 42);
  WaitForMaintenance();
  iree_hal_pool_query_stats(backing_pool, &stats);
  EXPECT_EQ(stats.bytes_committed, 0u);
  iree_hal_pool_release(backing_pool);
  iree_hal_slab_provider_release(provider);
  iree_async_notification_release(notification);
  iree_async_frontier_tracker_release(tracker);
}

TEST(TLSFPool, TrimRetainsSlabWithTaintedDeathFrontier) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_slab_provider_t* slab_provider = NULL;
  IREE_ASSERT_OK(iree_hal_cpu_slab_provider_create(/*min_alignment=*/0,
                                                   allocator, &slab_provider));
  iree_async_notification_t* notification = NULL;
  IREE_ASSERT_OK(iree_async_notification_create(
      test_proactor(), IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification));

  iree_hal_tlsf_pool_options_t options = DefaultOptions();
  options.tlsf_options.frontier_capacity = 1;
  iree_hal_test_epoch_query_t query = {
      /*.axis=*/TestQueueAxis(0),
      /*.completed_epoch=*/UINT64_MAX,
      /*.query_count=*/0,
  };
  iree_hal_pool_t* pool = NULL;
  IREE_ASSERT_OK(CreateTLSFPool(options, slab_provider, notification,
                                test_frontier_tracker(),
                                (iree_hal_pool_epoch_query_t){
                                    /*.fn=*/iree_hal_test_epoch_query,
                                    /*.user_data=*/&query,
                                },
                                allocator, &pool));

  iree_hal_pool_reservation_t reservations[2];
  iree_hal_pool_acquire_info_t acquire_info;
  iree_hal_pool_acquire_result_t result;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(reservations); ++i) {
    IREE_ASSERT_OK(
        AcquireOneReservation(pool, 2048, 16, /*requester_frontier=*/NULL,
                              IREE_HAL_POOL_RESERVE_FLAG_NONE, &reservations[i],
                              &acquire_info, &result));
  }

  MAKE_FRONTIER(first_frontier, 1, E(TestQueueAxis(0), 1));
  MAKE_FRONTIER(second_frontier, 1, E(TestQueueAxis(1), 1));
  ReleaseOneReservation(pool, &reservations[0], first_frontier);
  ReleaseOneReservation(pool, &reservations[1], second_frontier);
  iree_hal_pool_trim(pool, IREE_HAL_POOL_TRIM_FLAG_ALL, 0);

  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool, &stats);
  EXPECT_EQ(stats.slab_count, 1u);
  EXPECT_EQ(stats.bytes_committed, 4096u);
  EXPECT_EQ(query.query_count, 0u);

  iree_hal_pool_release(pool);
  iree_async_notification_release(notification);
  iree_hal_slab_provider_release(slab_provider);
}

TEST(TLSFPool, ReleaseNodeReuseAvoidsRepeatedHostAllocation) {
  iree_hal_test_counting_allocator_t allocator_state = {
      .backing_allocator = iree_allocator_system(),
      .allocation_call_count = 0,
      .free_call_count = 0,
  };
  iree_allocator_t allocator =
      iree_hal_test_counting_allocator(&allocator_state);

  iree_hal_slab_provider_t* slab_provider = NULL;
  IREE_ASSERT_OK(iree_hal_cpu_slab_provider_create(/*min_alignment=*/0,
                                                   allocator, &slab_provider));
  iree_async_notification_t* notification = NULL;
  IREE_ASSERT_OK(iree_async_notification_create(
      test_proactor(), IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification));

  iree_hal_pool_t* backing_pool = nullptr;
  iree_hal_pool_t* pool = NULL;
  IREE_ASSERT_OK(CreateFiniteTLSFPool(DefaultOptions(), slab_provider,
                                      notification, test_frontier_tracker(),
                                      allocator, &backing_pool, &pool));

  const iree_host_size_t allocation_call_count_after_create =
      allocator_state.allocation_call_count;

  iree_hal_pool_reservation_t reservation;
  iree_hal_pool_acquire_info_t reserve_info;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(AcquireOneReservation(
      pool, 128, 16, /*requester_frontier=*/NULL,
      IREE_HAL_POOL_RESERVE_FLAG_NONE, &reservation, &reserve_info, &result));
  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);
  EXPECT_GT(allocator_state.allocation_call_count,
            allocation_call_count_after_create);

  const iree_host_size_t allocation_call_count_after_first_reserve =
      allocator_state.allocation_call_count;
  ReleaseOneReservation(pool, &reservation, NULL);

  IREE_ASSERT_OK(AcquireOneReservation(
      pool, 128, 16, /*requester_frontier=*/NULL,
      IREE_HAL_POOL_RESERVE_FLAG_NONE, &reservation, &reserve_info, &result));
  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);
  EXPECT_EQ(allocator_state.allocation_call_count,
            allocation_call_count_after_first_reserve);

  ReleaseOneReservation(pool, &reservation, NULL);
  const iree_host_size_t free_call_count_before_trim =
      allocator_state.free_call_count;
  iree_hal_pool_trim(pool, IREE_HAL_POOL_TRIM_FLAG_EXCESS,
                     /*min_bytes_to_keep=*/0);
  EXPECT_GT(allocator_state.free_call_count, free_call_count_before_trim);

  iree_hal_pool_release(pool);
  iree_hal_pool_release(backing_pool);
  iree_async_notification_release(notification);
  iree_hal_slab_provider_release(slab_provider);
}

TEST_F(FiniteTLSFPoolTest, ReserveReusesDominatedFrontier) {
  iree_hal_pool_reservation_t reservation;
  iree_hal_pool_acquire_info_t reserve_info;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(AcquireOneReservation(
      pool_, 256, 16, /*requester_frontier=*/NULL,
      IREE_HAL_POOL_RESERVE_FLAG_NONE, &reservation, &reserve_info, &result));
  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);

  MAKE_FRONTIER(death, 1, E(TestQueueAxis(0), 10));
  ReleaseOneReservation(pool_, &reservation, death);

  MAKE_FRONTIER(requester, 1, E(TestQueueAxis(0), 10));
  IREE_ASSERT_OK(AcquireOneReservation(pool_, 256, 16, requester,
                                       IREE_HAL_POOL_RESERVE_FLAG_NONE,
                                       &reservation, &reserve_info, &result));
  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK);
  EXPECT_EQ(reservation.offset, 0u);
  ASSERT_NE(reserve_info.reuse_frontier, nullptr);
  EXPECT_EQ(reserve_info.reuse_frontier->entry_count, 1u);
  EXPECT_EQ(reserve_info.reuse_frontier->entries[0].axis, TestQueueAxis(0));
  EXPECT_EQ(reserve_info.reuse_frontier->entries[0].epoch, 10u);

  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool_, &stats);
  EXPECT_EQ(stats.reuse_count, 1u);
  EXPECT_EQ(stats.fresh_count, 1u);

  ReleaseOneReservation(pool_, &reservation, reserve_info.reuse_frontier);
  IREE_ASSERT_OK(AcquireOneReservation(pool_, 256, 16, requester,
                                       IREE_HAL_POOL_RESERVE_FLAG_NONE,
                                       &reservation, &reserve_info, &result));
  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK);
  ASSERT_NE(reserve_info.reuse_frontier, nullptr);
  EXPECT_EQ(reserve_info.reuse_frontier->entries[0].epoch, 10u);
  ReleaseOneReservation(pool_, &reservation, NULL);
}

TEST_F(FiniteTLSFPoolTest, ReuseFrontierSurvivesMetadataGrowth) {
  iree_hal_pool_reservation_t whole_range;
  iree_hal_pool_acquire_info_t whole_info;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(AcquireOneReservation(
      pool_, 4096, 16, /*requester_frontier=*/NULL,
      IREE_HAL_POOL_RESERVE_FLAG_NONE, &whole_range, &whole_info, &result));
  MAKE_FRONTIER(death, 2, E(TestQueueAxis(0), 10), E(TestQueueAxis(1), 20));
  const auto* backing = ReservationBacking(pool_, whole_range);
  ReleaseOneReservation(pool_, &whole_range, death);

  // More live ranges than the initial TLSF metadata capacity force metadata
  // growth while previously returned reuse frontiers are still borrowed.
  iree_hal_pool_reservation_t reservations[48];
  iree_hal_pool_acquire_info_t infos[IREE_ARRAYSIZE(reservations)];
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(reservations); ++i) {
    IREE_ASSERT_OK(AcquireOneReservation(pool_, 64, 16, death,
                                         IREE_HAL_POOL_RESERVE_FLAG_NONE,
                                         &reservations[i], &infos[i], &result));
    EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK);
    EXPECT_EQ(ReservationBacking(pool_, reservations[i]), backing);
  }
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(reservations); ++i) {
    ASSERT_NE(infos[i].reuse_frontier, nullptr);
    EXPECT_EQ(infos[i].reuse_frontier->entry_count, 2u);
    EXPECT_EQ(infos[i].reuse_frontier->entries[0].axis, TestQueueAxis(0));
    EXPECT_EQ(infos[i].reuse_frontier->entries[0].epoch, 10u);
    EXPECT_EQ(infos[i].reuse_frontier->entries[1].axis, TestQueueAxis(1));
    EXPECT_EQ(infos[i].reuse_frontier->entries[1].epoch, 20u);
    ReleaseOneReservation(pool_, &reservations[i], infos[i].reuse_frontier);
  }

  IREE_ASSERT_OK(
      AcquireOneReservation(pool_, 4096, 16, /*requester_frontier=*/NULL,
                            IREE_HAL_POOL_RESERVE_FLAG_DISALLOW_GROWTH,
                            &whole_range, &whole_info, &result));
  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_EXHAUSTED);
  IREE_ASSERT_OK(AcquireOneReservation(
      pool_, 4096, 16, death, IREE_HAL_POOL_RESERVE_FLAG_DISALLOW_GROWTH,
      &whole_range, &whole_info, &result));
  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK);
  ReleaseOneReservation(pool_, &whole_range, NULL);
}

TEST(TLSFPool, SplitRangesRetainReadinessUntilCompletion) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_slab_provider_t* slab_provider = NULL;
  IREE_ASSERT_OK(iree_hal_cpu_slab_provider_create(/*min_alignment=*/0,
                                                   allocator, &slab_provider));
  iree_async_notification_t* notification = NULL;
  IREE_ASSERT_OK(iree_async_notification_create(
      test_proactor(), IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification));

  iree_hal_test_epoch_query_t epoch_query = {TestQueueAxis(0), 0, 0};
  auto options = DefaultOptions();
  options.tlsf_options.range_length = 1040;
  iree_hal_pool_t* pool = NULL;
  IREE_ASSERT_OK(CreateTLSFPool(options, slab_provider, notification,
                                test_frontier_tracker(),
                                iree_hal_pool_epoch_query_t{
                                    /*.fn=*/iree_hal_test_epoch_query,
                                    /*.user_data=*/&epoch_query,
                                },
                                allocator, &pool));

  iree_hal_pool_reservation_t whole_range;
  iree_hal_pool_acquire_info_t info;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(AcquireOneReservation(
      pool, 1024, 16, /*requester_frontier=*/NULL,
      IREE_HAL_POOL_RESERVE_FLAG_NONE, &whole_range, &info, &result));
  ASSERT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);
  iree_hal_pool_reservation_t held;
  IREE_ASSERT_OK(AcquireOneReservation(pool, 16, 16, nullptr,
                                       IREE_HAL_POOL_RESERVE_FLAG_NONE, &held,
                                       &info, &result));
  MAKE_FRONTIER(death, 1, E(TestQueueAxis(0), 10));
  ReleaseOneReservation(pool, &whole_range, death);

  // A requester ordered after the previous use can reuse a subrange before
  // global completion. This does not establish readiness for other requesters.
  iree_hal_pool_reservation_t prefix;
  IREE_ASSERT_OK(AcquireOneReservation(
      pool, 256, 16, death, IREE_HAL_POOL_RESERVE_FLAG_DISALLOW_GROWTH, &prefix,
      &info, &result));
  ASSERT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK);
  EXPECT_EQ(prefix.offset, 0u);

  MAKE_FRONTIER(other_requester, 1, E(TestQueueAxis(1), 1));
  iree_hal_pool_reservation_t remainder;
  IREE_ASSERT_OK(AcquireOneReservation(
      pool, 256, 16, other_requester,
      IREE_HAL_POOL_RESERVE_FLAG_DISALLOW_GROWTH, &remainder, &info, &result));
  ASSERT_EQ(result, IREE_HAL_POOL_ACQUIRE_EXHAUSTED);
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool, &stats);
  EXPECT_EQ(stats.slab_count, 1u);
  EXPECT_EQ(stats.reservation_count, 2u);
  EXPECT_EQ(stats.bytes_reserved, 272u);

  // Completion makes the remaining range available to the other requester.
  epoch_query.completed_epoch = 10;
  IREE_ASSERT_OK(AcquireOneReservation(pool, 256, 16, other_requester,
                                       IREE_HAL_POOL_RESERVE_FLAG_NONE,
                                       &remainder, &info, &result));
  ASSERT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK);
  EXPECT_EQ(ReservationBacking(pool, remainder),
            ReservationBacking(pool, prefix));
  EXPECT_EQ(remainder.offset, 256u);

  ReleaseOneReservation(pool, &remainder, NULL);
  ReleaseOneReservation(pool, &prefix, NULL);
  ReleaseOneReservation(pool, &held, nullptr);
  iree_hal_pool_release(pool);
  iree_async_notification_release(notification);
  iree_hal_slab_provider_release(slab_provider);
}

TEST(TLSFPool, ReserveSkipsStaleHeadAndReturnsFreshLaterBlock) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_slab_provider_t* slab_provider = NULL;
  IREE_ASSERT_OK(iree_hal_cpu_slab_provider_create(/*min_alignment=*/0,
                                                   allocator, &slab_provider));
  iree_async_notification_t* notification = NULL;
  IREE_ASSERT_OK(iree_async_notification_create(
      test_proactor(), IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification));

  iree_hal_tlsf_pool_options_t options = {};
  options.tlsf_options.range_length = 1024;
  options.tlsf_options.alignment = 16;
  options.tlsf_options.initial_block_capacity = 16;
  options.tlsf_options.frontier_capacity = 2;

  iree_hal_pool_t* pool = NULL;
  IREE_ASSERT_OK(CreateTLSFPool(
      options, slab_provider, notification, test_frontier_tracker(),
      iree_hal_pool_epoch_query_null(), allocator, &pool));

  iree_hal_pool_reservation_t left_stale;
  iree_hal_pool_reservation_t middle_live;
  iree_hal_pool_reservation_t right_fresh;
  iree_hal_pool_reservation_t tail_live;
  iree_hal_pool_acquire_info_t reserve_info;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(AcquireOneReservation(
      pool, 256, 16, /*requester_frontier=*/NULL,
      IREE_HAL_POOL_RESERVE_FLAG_NONE, &left_stale, &reserve_info, &result));
  IREE_ASSERT_OK(AcquireOneReservation(
      pool, 256, 16, /*requester_frontier=*/NULL,
      IREE_HAL_POOL_RESERVE_FLAG_NONE, &middle_live, &reserve_info, &result));
  IREE_ASSERT_OK(AcquireOneReservation(
      pool, 256, 16, /*requester_frontier=*/NULL,
      IREE_HAL_POOL_RESERVE_FLAG_NONE, &right_fresh, &reserve_info, &result));
  IREE_ASSERT_OK(AcquireOneReservation(
      pool, 256, 16, /*requester_frontier=*/NULL,
      IREE_HAL_POOL_RESERVE_FLAG_NONE, &tail_live, &reserve_info, &result));
  EXPECT_EQ(left_stale.offset, 0u);
  EXPECT_EQ(right_fresh.offset, 512u);

  // Publish the stale block last so it becomes the bin head regardless of
  // whether asynchronous maintenance drains the two releases together.
  ReleaseOneReservation(pool, &right_fresh, NULL);
  WaitForMaintenance();
  MAKE_FRONTIER(death, 1, E(TestQueueAxis(0), 20));
  ReleaseOneReservation(pool, &left_stale, death);
  WaitForMaintenance();

  MAKE_FRONTIER(requester, 1, E(TestQueueAxis(0), 10));
  iree_hal_pool_reservation_t reservation;
  IREE_ASSERT_OK(AcquireOneReservation(
      pool, 256, 16, requester, IREE_HAL_POOL_RESERVE_FLAG_ALLOW_WAIT_FRONTIER,
      &reservation, &reserve_info, &result));
  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);
  EXPECT_EQ(reservation.offset, 512u);

  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool, &stats);
  EXPECT_EQ(stats.reuse_miss_count, 1u);
  EXPECT_EQ(stats.wait_count, 0u);

  MAKE_FRONTIER(dominating_requester, 1, E(TestQueueAxis(0), 20));
  iree_hal_pool_reservation_t stale_reservation;
  IREE_ASSERT_OK(AcquireOneReservation(
      pool, 256, 16, dominating_requester, IREE_HAL_POOL_RESERVE_FLAG_NONE,
      &stale_reservation, &reserve_info, &result));
  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK);
  EXPECT_EQ(stale_reservation.offset, 0u);

  ReleaseOneReservation(pool, &reservation, NULL);
  ReleaseOneReservation(pool, &stale_reservation, NULL);
  ReleaseOneReservation(pool, &middle_live, NULL);
  ReleaseOneReservation(pool, &tail_live, NULL);
  iree_hal_pool_release(pool);
  iree_async_notification_release(notification);
  iree_hal_slab_provider_release(slab_provider);
}

TEST(TLSFPool, ReserveGrowsInsteadOfWaitingForStaleBlock) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_slab_provider_t* slab_provider = NULL;
  IREE_ASSERT_OK(iree_hal_cpu_slab_provider_create(/*min_alignment=*/0,
                                                   allocator, &slab_provider));
  iree_async_notification_t* notification = NULL;
  IREE_ASSERT_OK(iree_async_notification_create(
      test_proactor(), IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification));

  iree_hal_tlsf_pool_options_t options = DefaultOptions();
  options.tlsf_options.range_length = 272;

  iree_hal_pool_t* pool = NULL;
  IREE_ASSERT_OK(CreateTLSFPool(
      options, slab_provider, notification, test_frontier_tracker(),
      iree_hal_pool_epoch_query_null(), allocator, &pool));

  iree_hal_pool_reservation_t reservation;
  iree_hal_pool_acquire_info_t reserve_info;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(AcquireOneReservation(
      pool, 256, 16, /*requester_frontier=*/NULL,
      IREE_HAL_POOL_RESERVE_FLAG_NONE, &reservation, &reserve_info, &result));
  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);

  const auto* first_backing = ReservationBacking(pool, reservation);
  iree_hal_pool_reservation_t held;
  IREE_ASSERT_OK(AcquireOneReservation(pool, 16, 16, nullptr,
                                       IREE_HAL_POOL_RESERVE_FLAG_NONE, &held,
                                       &reserve_info, &result));
  MAKE_FRONTIER(death, 1, E(TestQueueAxis(0), 20));
  ReleaseOneReservation(pool, &reservation, death);

  MAKE_FRONTIER(requester, 1, E(TestQueueAxis(0), 10));
  iree_hal_pool_reservation_t grown_reservation;
  IREE_ASSERT_OK(AcquireOneReservation(
      pool, 256, 16, requester, IREE_HAL_POOL_RESERVE_FLAG_NONE,
      &grown_reservation, &reserve_info, &result));
  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);
  EXPECT_EQ(reserve_info.reuse_frontier, nullptr);
  EXPECT_NE(ReservationBacking(pool, grown_reservation), first_backing);

  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool, &stats);
  // The retry rechecks old capacity after native growth preparation.
  EXPECT_EQ(stats.reuse_miss_count, 2u);
  EXPECT_EQ(stats.wait_count, 0u);

  MAKE_FRONTIER(dominating_requester, 1, E(TestQueueAxis(0), 20));
  IREE_ASSERT_OK(AcquireOneReservation(pool, 256, 16, dominating_requester,
                                       IREE_HAL_POOL_RESERVE_FLAG_NONE,
                                       &reservation, &reserve_info, &result));
  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK);
  EXPECT_EQ(reservation.offset, 0u);

  ReleaseOneReservation(pool, &reservation, NULL);
  ReleaseOneReservation(pool, &grown_reservation, NULL);
  ReleaseOneReservation(pool, &held, nullptr);
  iree_hal_pool_release(pool);
  iree_async_notification_release(notification);
  iree_hal_slab_provider_release(slab_provider);
}

TEST(TLSFPool, ReserveCanReportGrowthRequiredWithoutGrowing) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_slab_provider_t* slab_provider = NULL;
  IREE_ASSERT_OK(iree_hal_cpu_slab_provider_create(/*min_alignment=*/0,
                                                   allocator, &slab_provider));
  iree_async_notification_t* notification = NULL;
  IREE_ASSERT_OK(iree_async_notification_create(
      test_proactor(), IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification));

  iree_hal_tlsf_pool_options_t options = DefaultOptions();
  options.tlsf_options.range_length = 272;

  iree_hal_pool_t* pool = NULL;
  IREE_ASSERT_OK(CreateTLSFPool(
      options, slab_provider, notification, test_frontier_tracker(),
      iree_hal_pool_epoch_query_null(), allocator, &pool));

  iree_hal_pool_reservation_t reservation;
  iree_hal_pool_acquire_info_t reserve_info;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(AcquireOneReservation(
      pool, 256, 16, /*requester_frontier=*/NULL,
      IREE_HAL_POOL_RESERVE_FLAG_NONE, &reservation, &reserve_info, &result));
  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);

  const auto* first_backing = ReservationBacking(pool, reservation);
  iree_hal_pool_reservation_t held;
  IREE_ASSERT_OK(AcquireOneReservation(pool, 16, 16, nullptr,
                                       IREE_HAL_POOL_RESERVE_FLAG_NONE, &held,
                                       &reserve_info, &result));
  MAKE_FRONTIER(death, 1, E(TestQueueAxis(0), 20));
  ReleaseOneReservation(pool, &reservation, death);

  MAKE_FRONTIER(requester, 1, E(TestQueueAxis(0), 10));
  IREE_ASSERT_OK(AcquireOneReservation(
      pool, 256, 16, requester, IREE_HAL_POOL_RESERVE_FLAG_DISALLOW_GROWTH,
      &reservation, &reserve_info, &result));
  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_EXHAUSTED);
  EXPECT_TRUE(iree_all_bits_set(reserve_info.flags,
                                IREE_HAL_POOL_ACQUIRE_FLAG_GROWTH_REQUIRED));

  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool, &stats);
  EXPECT_EQ(stats.slab_count, 1u);
  EXPECT_EQ(stats.reuse_miss_count, 1u);
  EXPECT_EQ(stats.exhausted_count, 1u);
  EXPECT_EQ(stats.wait_count, 0u);

  iree_hal_pool_reservation_t grown_reservation;
  IREE_ASSERT_OK(AcquireOneReservation(
      pool, 256, 16, requester, IREE_HAL_POOL_RESERVE_FLAG_NONE,
      &grown_reservation, &reserve_info, &result));
  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);
  EXPECT_FALSE(iree_all_bits_set(reserve_info.flags,
                                 IREE_HAL_POOL_ACQUIRE_FLAG_GROWTH_REQUIRED));
  EXPECT_NE(ReservationBacking(pool, grown_reservation), first_backing);

  ReleaseOneReservation(pool, &grown_reservation, NULL);
  ReleaseOneReservation(pool, &held, nullptr);
  iree_hal_pool_release(pool);
  iree_async_notification_release(notification);
  iree_hal_slab_provider_release(slab_provider);
}

TEST(TLSFPool, ReserveRejectedTaintRemainsRejected) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_slab_provider_t* slab_provider = NULL;
  IREE_ASSERT_OK(iree_hal_cpu_slab_provider_create(/*min_alignment=*/0,
                                                   allocator, &slab_provider));
  iree_async_notification_t* notification = NULL;
  IREE_ASSERT_OK(iree_async_notification_create(
      test_proactor(), IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification));

  iree_hal_tlsf_pool_options_t options = DefaultOptions();
  options.tlsf_options.range_length = 256;
  options.tlsf_options.frontier_capacity = 1;

  iree_hal_pool_t* pool = NULL;
  IREE_ASSERT_OK(CreateTLSFPool(
      options, slab_provider, notification, test_frontier_tracker(),
      iree_hal_pool_epoch_query_null(), allocator, &pool));

  iree_hal_pool_reservation_t reservation;
  iree_hal_pool_acquire_info_t reserve_info;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(AcquireOneReservation(
      pool, 256, 16, /*requester_frontier=*/NULL,
      IREE_HAL_POOL_RESERVE_FLAG_NONE, &reservation, &reserve_info, &result));
  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);

  MAKE_FRONTIER(oversized_death, 2, E(TestQueueAxis(0), 10),
                E(TestQueueAxis(1), 20));
  const auto* first_backing = ReservationBacking(pool, reservation);
  ReleaseOneReservation(pool, &reservation, oversized_death);

  MAKE_FRONTIER(requester, 2, E(TestQueueAxis(0), 100),
                E(TestQueueAxis(1), 100));
  iree_hal_pool_reservation_t first_grown_reservation;
  IREE_ASSERT_OK(AcquireOneReservation(
      pool, 256, 16, requester, IREE_HAL_POOL_RESERVE_FLAG_ALLOW_WAIT_FRONTIER,
      &first_grown_reservation, &reserve_info, &result));
  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);
  EXPECT_NE(ReservationBacking(pool, first_grown_reservation), first_backing);

  iree_hal_pool_reservation_t second_grown_reservation;
  IREE_ASSERT_OK(AcquireOneReservation(
      pool, 256, 16, requester, IREE_HAL_POOL_RESERVE_FLAG_ALLOW_WAIT_FRONTIER,
      &second_grown_reservation, &reserve_info, &result));
  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);

  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool, &stats);
  // Each growth retry rechecks that the original tainted range stays unusable.
  EXPECT_EQ(stats.reuse_miss_count, 4u);
  EXPECT_EQ(stats.exhausted_count, 0u);
  EXPECT_EQ(stats.wait_count, 0u);

  ReleaseOneReservation(pool, &second_grown_reservation, NULL);
  ReleaseOneReservation(pool, &first_grown_reservation, NULL);
  iree_hal_pool_release(pool);
  iree_async_notification_release(notification);
  iree_hal_slab_provider_release(slab_provider);
}

TEST_F(TLSFPoolTest, WrapReservationCreatesBuffer) {
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
  EXPECT_EQ(iree_hal_buffer_allocation_size(buffer), 4096u);
  EXPECT_EQ(iree_hal_buffer_byte_length(buffer), 128u);

  iree_hal_buffer_mapping_t mapping;
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      buffer, IREE_HAL_MAPPING_MODE_SCOPED,
      IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
      IREE_HAL_BUFFER_MAP_FLAG_NONE, 0, 128, &mapping));
  memset(mapping.contents.data, 0x7C, 128);
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));

  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool_, &stats);
  EXPECT_EQ(stats.reservation_count, 1u);

  iree_hal_buffer_release(buffer);

  iree_hal_pool_query_stats(pool_, &stats);
  EXPECT_EQ(stats.reservation_count, 0u);
  EXPECT_EQ(stats.release_count, 1u);
}

TEST_F(TLSFPoolTest, BorrowedMaterializationDoesNotReleaseReservation) {
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

TEST(TLSFPool, BackingMaterializationFailureReturnsReservation) {
  iree_hal_slab_provider_t* provider = nullptr;
  IREE_ASSERT_OK(iree_hal_test_opaque_slab_provider_create(
      iree_allocator_system(), &provider));
  iree_async_notification_t* notification = nullptr;
  IREE_ASSERT_OK(iree_async_notification_create(
      test_proactor(), IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification));
  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(CreateTLSFPool(DefaultOptions(), provider, notification,
                                test_frontier_tracker(), {},
                                iree_allocator_system(), &pool));
  auto* native =
      reinterpret_cast<iree_hal_test_opaque_slab_provider_t*>(provider);
  native->fail_wrap = true;
  iree_hal_pool_reservation_t unchanged;
  memset(&unchanged, 0xA5, sizeof(unchanged));
  auto reservation = unchanged;
  iree_hal_pool_acquire_info_t info;
  iree_hal_pool_acquire_result_t result;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        AcquireOneReservation(pool, 256, 16, nullptr,
                                              IREE_HAL_POOL_RESERVE_FLAG_NONE,
                                              &reservation, &info, &result));
  EXPECT_EQ(memcmp(&reservation, &unchanged, sizeof(reservation)), 0);
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool, &stats);
  EXPECT_EQ(stats.bytes_committed, 0u);
  EXPECT_EQ(stats.reservation_count, 0u);
  native->fail_wrap = false;
  IREE_ASSERT_OK(AcquireOneReservation(pool, 256, 16, nullptr,
                                       IREE_HAL_POOL_RESERVE_FLAG_NONE,
                                       &reservation, &info, &result));
  ASSERT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);
  iree_hal_pool_release_reservations(pool, 1, &reservation, nullptr);
  iree_hal_pool_release(pool);
  iree_hal_slab_provider_release(provider);
  iree_async_notification_release(notification);
}

TEST(TLSFPool, UsesProviderHooks) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_slab_provider_t* slab_provider = NULL;
  IREE_ASSERT_OK(
      iree_hal_test_opaque_slab_provider_create(allocator, &slab_provider));
  iree_async_notification_t* notification = NULL;
  IREE_ASSERT_OK(iree_async_notification_create(
      test_proactor(), IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification));

  iree_hal_pool_t* backing_pool = nullptr;
  IREE_ASSERT_OK(CreateNativePool(slab_provider, notification,
                                  test_frontier_tracker(), {}, &backing_pool));
  const auto options = DefaultOptions();
  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(
      iree_hal_tlsf_pool_create(backing_pool, &options, allocator, &pool));

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

  // Local trim preserves live views and leaves native retention policy alone.
  auto* provider = (iree_hal_test_opaque_slab_provider_t*)slab_provider;
  const iree_hal_pool_trim_flags_t trim_flags[] = {
      IREE_HAL_POOL_TRIM_FLAG_NONE,
      IREE_HAL_POOL_TRIM_FLAG_EXCESS,
      IREE_HAL_POOL_TRIM_FLAG_ALL,
      IREE_HAL_POOL_TRIM_FLAG_ALL | IREE_HAL_POOL_TRIM_FLAG_EXCESS,
  };
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(trim_flags); ++i) {
    iree_hal_pool_trim(pool, trim_flags[i], /*min_bytes_to_keep=*/0);
    EXPECT_EQ(provider->trim_count, 0u);
    iree_hal_pool_stats_t stats;
    iree_hal_pool_query_stats(pool, &stats);
    EXPECT_EQ(stats.reservation_count, 1u);
  }

  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(trim_flags); ++i) {
    iree_hal_pool_trim(backing_pool, trim_flags[i], 0);
    EXPECT_EQ(provider->trim_count, i + 1);
    EXPECT_EQ(provider->last_trim_flags, trim_flags[i]);
  }

  iree_hal_buffer_mapping_t mapping;
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      buffer, IREE_HAL_MAPPING_MODE_SCOPED,
      IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
      IREE_HAL_BUFFER_MAP_FLAG_NONE, 0, 128, &mapping));
  memset(mapping.contents.data, 0x4D, 128);
  EXPECT_EQ(((uint8_t*)mapping.contents.data)[127], 0x4D);
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

TEST(TLSFPool, CreateRejectsASANWhenProviderCannotAdviseRanges) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_slab_provider_t* slab_provider = NULL;
  IREE_ASSERT_OK(iree_hal_cpu_slab_provider_create(/*min_alignment=*/0,
                                                   allocator, &slab_provider));
  iree_async_notification_t* notification = NULL;
  IREE_ASSERT_OK(iree_async_notification_create(
      test_proactor(), IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification));

  iree_hal_tlsf_pool_options_t options = DefaultOptions();
  options.asan = ShadowOptions();

  iree_hal_pool_t* pool = NULL;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      CreateTLSFPool(options, slab_provider, notification,
                     test_frontier_tracker(), iree_hal_pool_epoch_query_null(),
                     allocator, &pool));
  EXPECT_EQ(pool, nullptr);

  iree_async_notification_release(notification);
  iree_hal_slab_provider_release(slab_provider);
}

TEST(TLSFPool, ASANAdvisesBackingRangeAndExposesUserRange) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_slab_provider_t* slab_provider = NULL;
  IREE_ASSERT_OK(
      iree_hal_test_opaque_slab_provider_create(allocator, &slab_provider));
  iree_async_notification_t* notification = NULL;
  IREE_ASSERT_OK(iree_async_notification_create(
      test_proactor(), IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification));

  iree_hal_tlsf_pool_options_t options = DefaultOptions();
  options.tlsf_options.range_length = 512;
  options.tlsf_options.alignment = 16;
  options.asan = ShadowOptions();

  iree_hal_pool_t* pool = NULL;
  IREE_ASSERT_OK(CreateTLSFPool(
      options, slab_provider, notification, test_frontier_tracker(),
      iree_hal_pool_epoch_query_null(), allocator, &pool));

  iree_hal_pool_reservation_t reservation;
  iree_hal_pool_acquire_info_t reserve_info;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(AcquireOneReservation(
      pool, 13, 16, /*requester_frontier=*/NULL,
      IREE_HAL_POOL_RESERVE_FLAG_NONE, &reservation, &reserve_info, &result));

  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);
  EXPECT_EQ(reservation.offset, 64u);
  EXPECT_EQ(reservation.byte_length, 13u);

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
  EXPECT_EQ(provider->last_asan_layout.backing_length, 128u);
  EXPECT_EQ(provider->last_asan_layout.user_offset, 64u);
  EXPECT_EQ(provider->last_asan_layout.user_length, 13u);
  EXPECT_EQ(provider->last_asan_layout.right_redzone_length, 51u);

  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool, &stats);
  EXPECT_EQ(stats.bytes_reserved, 128u);
  EXPECT_EQ(stats.bytes_committed, 640u);
  EXPECT_EQ(stats.bytes_free, 512u);

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
  EXPECT_EQ(iree_hal_buffer_allocation_size(buffer), 640u);
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
  iree_async_notification_release(notification);
  iree_hal_slab_provider_release(slab_provider);
}

TEST(TLSFPool, MixedDedicatedBatchKeepsAdviceAtExecutionBoundaries) {
  const auto allocator = iree_allocator_system();
  iree_hal_slab_provider_t* slab_provider = nullptr;
  IREE_ASSERT_OK(
      iree_hal_test_opaque_slab_provider_create(allocator, &slab_provider));
  iree_async_notification_t* notification = nullptr;
  IREE_ASSERT_OK(iree_async_notification_create(
      test_proactor(), IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification));
  auto options = DefaultOptions();
  options.tlsf_options.range_length = 512;
  options.asan = ShadowOptions();
  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(CreateTLSFPool(
      options, slab_provider, notification, test_frontier_tracker(),
      iree_hal_pool_epoch_query_null(), allocator, &pool));
  iree_hal_pool_reservation_request_t requests[] = {
      MakeReservationRequest(13, 16), MakeReservationRequest(8193, 64),
      MakeReservationRequest(97, 16)};
  for (auto& request : requests) {
    request.params.usage |= IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
  }
  iree_hal_pool_reservation_t reservations[3];
  iree_hal_pool_acquire_info_t infos[3];
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations(
      pool, 3, requests, nullptr, IREE_HAL_POOL_RESERVE_FLAG_NONE, reservations,
      infos, &result));
  ASSERT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);
  auto* provider = (iree_hal_test_opaque_slab_provider_t*)slab_provider;
  EXPECT_EQ(
      iree_atomic_load(&provider->asan_advice_count, iree_memory_order_relaxed),
      0);
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool, &stats);
  EXPECT_EQ(stats.bytes_reserved, 128u + 8320u + 192u);
  EXPECT_EQ(stats.bytes_committed, 640u + 8320u);
  EXPECT_EQ(stats.slab_count, 2u);
  const iree_device_size_t backing_lengths[] = {128, 8320, 192};
  for (size_t i = 0; i < 3; ++i) {
    iree_hal_pool_advise_asan_reservations(
        pool, 1, &reservations[i], IREE_HAL_ASAN_RANGE_ADVICE_FLAG_ALLOCATED);
    EXPECT_EQ(provider->last_asan_layout.backing_length, backing_lengths[i]);
    EXPECT_EQ(provider->last_asan_layout.user_offset, 64u);
    EXPECT_EQ(provider->last_asan_layout.user_length,
              requests[i].allocation_size);
  }
  iree_hal_buffer_t* buffers[3];
  IREE_ASSERT_OK(iree_hal_pool_materialize_reservations(
      pool, 3, requests, reservations,
      IREE_HAL_POOL_MATERIALIZE_FLAG_TRANSFER_RESERVATION_OWNERSHIP, buffers));
  for (size_t i = 0; i < 3; ++i) {
    const uint8_t pattern = static_cast<uint8_t>(0x51 + i);
    IREE_ASSERT_OK(iree_hal_buffer_map_fill(
        buffers[i], 0, IREE_HAL_WHOLE_BUFFER, &pattern, sizeof(pattern)));
    uint8_t actual = 0;
    IREE_ASSERT_OK(iree_hal_buffer_map_read(
        buffers[i], requests[i].allocation_size - 1, &actual, sizeof(actual)));
    EXPECT_EQ(actual, pattern);
    iree_hal_buffer_release(buffers[i]);
  }
  EXPECT_EQ(iree_atomic_load(&provider->asan_released_count,
                             iree_memory_order_relaxed),
            3);
  iree_hal_pool_release(pool);
  iree_async_notification_release(notification);
  iree_hal_slab_provider_release(slab_provider);
}

TEST(TLSFPool, GuardedGrowingPoolReportsUsableSourceLimit) {
  const auto allocator = iree_allocator_system();
  iree_hal_slab_provider_t* slab_provider = nullptr;
  IREE_ASSERT_OK(
      iree_hal_test_opaque_slab_provider_create(allocator, &slab_provider));
  iree_async_notification_t* notification = nullptr;
  IREE_ASSERT_OK(iree_async_notification_create(
      test_proactor(), IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification));
  iree_hal_pool_t* source_pool = nullptr;
  IREE_ASSERT_OK(CreateTLSFPool(
      DefaultOptions(), slab_provider, notification, test_frontier_tracker(),
      iree_hal_pool_epoch_query_null(), allocator, &source_pool));
  const auto params = MakeReservationRequest(1024, 64).params;
  iree_hal_buffer_t* backing = nullptr;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      source_pool, params, 1024, iree_infinite_timeout(), &backing));
  iree_hal_fixed_block_pool_options_t parent_options = {.block_size = 1024,
                                                        .alignment = 64};
  iree_hal_pool_t* parent = nullptr;
  IREE_ASSERT_OK(iree_hal_fixed_block_pool_create_from_buffer(
      backing, 0, IREE_HAL_WHOLE_BUFFER, &parent_options, allocator, &parent));
  iree_hal_buffer_release(backing);
  auto options = DefaultOptions();
  options.tlsf_options.range_length = 128;
  options.asan = ShadowOptions();
  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(iree_hal_tlsf_pool_create(parent, &options, allocator, &pool));
  iree_hal_pool_capabilities_t capabilities;
  iree_hal_pool_query_capabilities(pool, &capabilities);
  EXPECT_EQ(capabilities.max_allocation_size, 944u);
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool, params, capabilities.max_allocation_size, iree_infinite_timeout(),
      &buffer));
  EXPECT_EQ(iree_hal_buffer_byte_length(buffer), 944u);
  iree_hal_buffer_release(buffer);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        iree_hal_pool_allocate_buffer(
                            pool, params, capabilities.max_allocation_size + 1,
                            iree_immediate_timeout(), &buffer));
  iree_hal_pool_release(pool);
  iree_hal_pool_release(parent);
  iree_hal_pool_release(source_pool);
  iree_async_notification_release(notification);
  iree_hal_slab_provider_release(slab_provider);
}

TEST(TLSFPool, FailedBatchDoesNotApplyNativeAdvice) {
  iree_hal_slab_provider_t* slab_provider = nullptr;
  IREE_ASSERT_OK(iree_hal_test_opaque_slab_provider_create(
      iree_allocator_system(), &slab_provider));
  iree_async_notification_t* notification = nullptr;
  IREE_ASSERT_OK(iree_async_notification_create(
      test_proactor(), IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification));
  auto options = DefaultOptions();
  options.asan = ShadowOptions();
  options.budget_limit = 192;
  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(CreateTLSFPool(
      options, slab_provider, notification, test_frontier_tracker(),
      iree_hal_pool_epoch_query_null(), iree_allocator_system(), &pool));
  const iree_hal_pool_reservation_request_t requests[] = {
      MakeReservationRequest(13, 16),
      MakeReservationRequest(13, 16),
  };
  iree_hal_pool_reservation_t reservations[2];
  iree_hal_pool_acquire_info_t infos[2];
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations(
      pool, 2, requests, nullptr, IREE_HAL_POOL_RESERVE_FLAG_NONE, reservations,
      infos, &result));
  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_OVER_BUDGET);
  const auto* provider =
      reinterpret_cast<iree_hal_test_opaque_slab_provider_t*>(slab_provider);
  EXPECT_EQ(
      iree_atomic_load(&provider->asan_advice_count, iree_memory_order_relaxed),
      0);
  iree_hal_pool_release(pool);
  iree_async_notification_release(notification);
  iree_hal_slab_provider_release(slab_provider);
}

TEST(TLSFPool, FinitePoolsPreserveNativeAdviceCoordinates) {
  auto allocator = iree_allocator_system();
  iree_hal_slab_provider_t* slab_provider = nullptr;
  IREE_ASSERT_OK(
      iree_hal_test_opaque_slab_provider_create(allocator, &slab_provider));
  iree_async_notification_t* notification = nullptr;
  IREE_ASSERT_OK(iree_async_notification_create(
      test_proactor(), IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification));
  iree_hal_pool_t* source_pool = nullptr;
  IREE_ASSERT_OK(CreateTLSFPool(
      DefaultOptions(), slab_provider, notification, test_frontier_tracker(),
      iree_hal_pool_epoch_query_null(), allocator, &source_pool));
  auto params = MakeReservationRequest(2048, 16).params;
  params.type |= IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
  params.usage |= IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
  iree_hal_buffer_t* source = nullptr;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      source_pool, params, 2048, iree_immediate_timeout(), &source));
  iree_hal_buffer_t* view = nullptr;
  IREE_ASSERT_OK(iree_hal_buffer_subspan(source, 128, 1024, allocator, &view));

  auto* provider = (iree_hal_test_opaque_slab_provider_t*)slab_provider;
  for (bool use_tlsf : {false, true}) {
    SCOPED_TRACE(use_tlsf ? "TLSF" : "fixed block");
    iree_hal_pool_t* child = nullptr;
    if (use_tlsf) {
      iree_hal_tlsf_pool_options_t options = {.asan = ShadowOptions()};
      IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                            iree_hal_tlsf_pool_create_from_buffer(
                                view, 64, 64, &options, allocator, &child));
      EXPECT_EQ(nullptr, child);
      IREE_ASSERT_OK(iree_hal_tlsf_pool_create_from_buffer(
          view, 64, 768, &options, allocator, &child));
      iree_hal_pool_capabilities_t capabilities;
      iree_hal_pool_query_capabilities(child, &capabilities);
      EXPECT_EQ(688u, capabilities.max_allocation_size);
      iree_hal_buffer_t* largest = nullptr;
      IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
          child, params, capabilities.max_allocation_size,
          iree_immediate_timeout(), &largest));
      iree_hal_buffer_release(largest);
      largest = source;
      IREE_EXPECT_STATUS_IS(
          IREE_STATUS_OUT_OF_RANGE,
          iree_hal_pool_allocate_buffer(child, params,
                                        capabilities.max_allocation_size + 1,
                                        iree_immediate_timeout(), &largest));
      EXPECT_EQ(source, largest);
    } else {
      iree_hal_fixed_block_pool_options_t options = {
          .block_size = 13, .alignment = 16, .asan = ShadowOptions()};
      IREE_ASSERT_OK(iree_hal_fixed_block_pool_create_from_buffer(
          view, 64, 768, &options, allocator, &child));
    }
    iree_hal_buffer_t* allocation = nullptr;
    IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
        child, params, 13, iree_immediate_timeout(), &allocation));
    EXPECT_EQ(192u, provider->last_asan_backing_offset);
    EXPECT_EQ(64u, provider->last_asan_layout.user_offset);
    EXPECT_EQ(13u, provider->last_asan_layout.user_length);
    EXPECT_EQ(128u, provider->last_asan_layout.backing_length);
    EXPECT_EQ(256u, iree_hal_buffer_memory_view(allocation).offset);
    const uint32_t value = 0x12345678;
    IREE_ASSERT_OK(
        iree_hal_buffer_map_write(allocation, 0, &value, sizeof(value)));
    uint32_t actual = 0;
    IREE_ASSERT_OK(
        iree_hal_buffer_map_read(source, 256, &actual, sizeof(actual)));
    EXPECT_EQ(value, actual);
    iree_hal_buffer_release(allocation);
    EXPECT_EQ(192u, provider->last_asan_backing_offset);
    EXPECT_GT(provider->last_asan_released_sequence,
              provider->last_asan_allocated_sequence);
    iree_hal_pool_trim(child, IREE_HAL_POOL_TRIM_FLAG_ALL, 0);
    EXPECT_EQ(0u, provider->trim_count);
    iree_hal_pool_release(child);
  }
  iree_hal_buffer_release(view);
  iree_hal_buffer_release(source);
  iree_hal_pool_release(source_pool);
  iree_async_notification_release(notification);
  iree_hal_slab_provider_release(slab_provider);
}

TEST(TLSFPool, GuardedFinitePoolsKeepPendingBookkeepingSideEffectFree) {
  for (bool use_tlsf : {false, true}) {
    SCOPED_TRACE(use_tlsf ? "TLSF" : "fixed block");
    auto allocator = iree_allocator_system();
    iree_async_frontier_tracker_t* tracker = nullptr;
    IREE_ASSERT_OK(iree_async_frontier_tracker_create(
        iree_async_frontier_tracker_options_default(), allocator, &tracker));
    const iree_async_axis_t axis = iree_async_axis_make_queue(1, 0, 0, 0, 0);
    IREE_ASSERT_OK(
        iree_async_frontier_tracker_register_axis(tracker, axis, nullptr));
    iree_async_single_frontier_t prior_use = {.entry_count = 1};
    prior_use.entries[0] = {axis, 1};
    const auto* frontier =
        iree_async_fixed_frontier_as_const_frontier(&prior_use);
    iree_async_notification_t* notification = nullptr;
    IREE_ASSERT_OK(iree_async_notification_create(
        test_proactor(), IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification));
    iree_hal_slab_provider_t* slab_provider = nullptr;
    IREE_ASSERT_OK(
        iree_hal_test_opaque_slab_provider_create(allocator, &slab_provider));
    auto* provider = (iree_hal_test_opaque_slab_provider_t*)slab_provider;
    iree_hal_pool_t* backing_pool = nullptr;
    iree_hal_pool_t* source_pool = nullptr;
    IREE_ASSERT_OK(CreateFiniteTLSFPool(DefaultOptions(), slab_provider,
                                        notification, tracker, allocator,
                                        &backing_pool, &source_pool));
    EXPECT_FALSE(iree_hal_pool_requires_asan_advice(source_pool));
    iree_hal_pool_reservation_t source_reservation;
    iree_hal_pool_acquire_info_t source_info;
    iree_hal_pool_acquire_result_t result;
    IREE_ASSERT_OK(AcquireOneReservation(
        source_pool, 4096, 16, nullptr, IREE_HAL_POOL_RESERVE_FLAG_NONE,
        &source_reservation, &source_info, &result));
    ASSERT_EQ(IREE_HAL_POOL_ACQUIRE_OK_FRESH, result);
    ReleaseOneReservation(source_pool, &source_reservation, frontier);
    IREE_ASSERT_OK(
        AcquireOneReservation(source_pool, 4096, 16, nullptr,
                              IREE_HAL_POOL_RESERVE_FLAG_ALLOW_WAIT_FRONTIER,
                              &source_reservation, &source_info, &result));
    ASSERT_EQ(IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT, result);
    auto params = MakeReservationRequest(4096, 16).params;
    params.usage |= IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
    iree_hal_buffer_t* source = nullptr;
    IREE_ASSERT_OK(MaterializeOneReservation(
        source_pool, params, &source_reservation,
        IREE_HAL_POOL_MATERIALIZE_FLAG_NONE, &source));

    iree_hal_pool_t* child = nullptr;
    if (use_tlsf) {
      iree_hal_tlsf_pool_options_t options = {};
      options.tlsf_options.frontier_capacity = 2;
      options.asan = ShadowOptions();
      IREE_ASSERT_OK(iree_hal_tlsf_pool_create_from_buffer(
          source, 0, 256, &options, allocator, &child));
    } else {
      iree_hal_fixed_block_pool_options_t options = {.block_size = 176,
                                                     .frontier_capacity = 2,
                                                     .alignment = 16,
                                                     .asan = ShadowOptions()};
      IREE_ASSERT_OK(iree_hal_fixed_block_pool_create_from_buffer(
          source, 0, 256, &options, allocator, &child));
    }
    EXPECT_TRUE(iree_hal_pool_requires_asan_advice(child));

    // Causal eligibility permits reservation and view creation before the
    // producer finishes, but it grants no authority to change shadow bytes.
    iree_hal_pool_reservation_t reservation;
    iree_hal_pool_acquire_info_t info;
    IREE_ASSERT_OK(AcquireOneReservation(child, 176, 16, frontier,
                                         IREE_HAL_POOL_RESERVE_FLAG_NONE,
                                         &reservation, &info, &result));
    ASSERT_EQ(IREE_HAL_POOL_ACQUIRE_OK, result);
    iree_hal_buffer_t* buffer = nullptr;
    IREE_ASSERT_OK(MaterializeOneReservation(
        child, params, &reservation, IREE_HAL_POOL_MATERIALIZE_FLAG_NONE,
        &buffer));
    EXPECT_EQ(0, iree_atomic_load(&provider->asan_advice_count,
                                  iree_memory_order_relaxed));
    iree_hal_buffer_release(buffer);
    ReleaseOneReservation(child, &reservation, info.reuse_frontier);
    EXPECT_EQ(0, iree_atomic_load(&provider->asan_advice_count,
                                  iree_memory_order_relaxed));

    // The synchronous helper's timed-out wait returns the exact prerequisite
    // without ever starting a new allocation lifetime.
    buffer = nullptr;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_DEADLINE_EXCEEDED,
        iree_hal_pool_allocate_buffer(child, params, 176,
                                      iree_immediate_timeout(), &buffer));
    EXPECT_EQ(nullptr, buffer);
    EXPECT_EQ(0, iree_atomic_load(&provider->asan_advice_count,
                                  iree_memory_order_relaxed));

    IREE_ASSERT_OK(AcquireOneReservation(
        child, 176, 16, nullptr, IREE_HAL_POOL_RESERVE_FLAG_ALLOW_WAIT_FRONTIER,
        &reservation, &info, &result));
    ASSERT_EQ(IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT, result);
    IREE_ASSERT_OK(MaterializeOneReservation(
        child, params, &reservation, IREE_HAL_POOL_MATERIALIZE_FLAG_NONE,
        &buffer));
    EXPECT_EQ(0, iree_atomic_load(&provider->asan_advice_count,
                                  iree_memory_order_relaxed));
    iree_async_frontier_tracker_advance(tracker, axis, 1);
    iree_hal_pool_advise_asan_reservations(
        child, 1, &reservation, IREE_HAL_ASAN_RANGE_ADVICE_FLAG_ALLOCATED);
    EXPECT_EQ(1, iree_atomic_load(&provider->asan_allocated_count,
                                  iree_memory_order_relaxed));
    const uint32_t value = 0x12345678;
    IREE_ASSERT_OK(iree_hal_buffer_map_write(buffer, 0, &value, sizeof(value)));
    uint32_t observed = 0;
    IREE_ASSERT_OK(
        iree_hal_buffer_map_read(buffer, 0, &observed, sizeof(observed)));
    EXPECT_EQ(value, observed);
    iree_hal_pool_advise_asan_reservations(
        child, 1, &reservation, IREE_HAL_ASAN_RANGE_ADVICE_FLAG_RELEASED);
    EXPECT_EQ(1, iree_atomic_load(&provider->asan_released_count,
                                  iree_memory_order_relaxed));
    EXPECT_LT(provider->last_asan_allocated_sequence,
              provider->last_asan_released_sequence);
    ReleaseOneReservation(child, &reservation, nullptr);
    iree_hal_buffer_release(buffer);
    EXPECT_EQ(2, iree_atomic_load(&provider->asan_advice_count,
                                  iree_memory_order_relaxed));
    iree_hal_pool_release(child);
    iree_hal_buffer_release(source);
    ReleaseOneReservation(source_pool, &source_reservation, nullptr);
    iree_hal_pool_release(source_pool);
    iree_hal_pool_release(backing_pool);
    iree_hal_slab_provider_release(slab_provider);
    iree_async_notification_release(notification);
    iree_async_frontier_tracker_release(tracker);
  }
}

TEST(TLSFPool, ASANQuarantineDelaysReleasedRangeReuse) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_slab_provider_t* slab_provider = NULL;
  IREE_ASSERT_OK(
      iree_hal_test_opaque_slab_provider_create(allocator, &slab_provider));
  iree_async_notification_t* notification = NULL;
  IREE_ASSERT_OK(iree_async_notification_create(
      test_proactor(), IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification));

  iree_hal_tlsf_pool_options_t options = DefaultOptions();
  options.tlsf_options.range_length = 512;
  options.tlsf_options.alignment = 16;
  options.asan = ShadowOptions();
  options.asan.quarantine_size = 128;

  iree_hal_pool_t* pool = NULL;
  IREE_ASSERT_OK(CreateTLSFPool(
      options, slab_provider, notification, test_frontier_tracker(),
      iree_hal_pool_epoch_query_null(), allocator, &pool));

  iree_hal_pool_reservation_t first_reservation;
  iree_hal_pool_acquire_info_t reserve_info;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(
      AcquireOneReservation(pool, 13, 16, /*requester_frontier=*/NULL,
                            IREE_HAL_POOL_RESERVE_FLAG_NONE, &first_reservation,
                            &reserve_info, &result));
  ASSERT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);
  EXPECT_EQ(first_reservation.offset, 64u);
  ReleaseOneReservation(pool, &first_reservation,
                        /*death_frontier=*/NULL);

  iree_hal_pool_reservation_t second_reservation;
  IREE_ASSERT_OK(
      AcquireOneReservation(pool, 13, 16, /*requester_frontier=*/NULL,
                            IREE_HAL_POOL_RESERVE_FLAG_NONE,
                            &second_reservation, &reserve_info, &result));
  ASSERT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);
  EXPECT_NE(second_reservation.offset, first_reservation.offset);
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool, &stats);
  EXPECT_EQ(stats.bytes_quarantined, 128u);
  EXPECT_EQ(stats.quarantine_eviction_count, 0u);
  EXPECT_EQ(stats.bytes_free, stats.bytes_committed - 256u);
  ReleaseOneReservation(pool, &second_reservation,
                        /*death_frontier=*/NULL);

  iree_hal_pool_reservation_t third_reservation;
  IREE_ASSERT_OK(
      AcquireOneReservation(pool, 13, 16, /*requester_frontier=*/NULL,
                            IREE_HAL_POOL_RESERVE_FLAG_NONE, &third_reservation,
                            &reserve_info, &result));
  ASSERT_TRUE(result == IREE_HAL_POOL_ACQUIRE_OK ||
              result == IREE_HAL_POOL_ACQUIRE_OK_FRESH);
  EXPECT_EQ(third_reservation.offset, first_reservation.offset);
  iree_hal_pool_query_stats(pool, &stats);
  EXPECT_EQ(stats.bytes_quarantined, 128u);
  EXPECT_EQ(stats.quarantine_eviction_count, 1u);
  EXPECT_EQ(stats.bytes_free, stats.bytes_committed - 256u);

  ReleaseOneReservation(pool, &third_reservation,
                        /*death_frontier=*/NULL);
  iree_hal_pool_trim(pool, IREE_HAL_POOL_TRIM_FLAG_NONE, 0);
  iree_hal_pool_query_stats(pool, &stats);
  EXPECT_EQ(stats.bytes_quarantined, 0u);
  EXPECT_EQ(stats.quarantine_eviction_count, 3u);
  iree_hal_pool_release(pool);
  iree_async_notification_release(notification);
  iree_hal_slab_provider_release(slab_provider);
}

TEST(TLSFPool, ASANZeroQuarantineReusesReleasedRangeImmediately) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_slab_provider_t* slab_provider = NULL;
  IREE_ASSERT_OK(
      iree_hal_test_opaque_slab_provider_create(allocator, &slab_provider));
  iree_async_notification_t* notification = NULL;
  IREE_ASSERT_OK(iree_async_notification_create(
      test_proactor(), IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification));

  iree_hal_tlsf_pool_options_t options = DefaultOptions();
  options.tlsf_options.range_length = 512;
  options.tlsf_options.alignment = 16;
  options.asan = ShadowOptions();
  options.asan.quarantine_size = 0;

  iree_hal_pool_t* pool = NULL;
  IREE_ASSERT_OK(CreateTLSFPool(
      options, slab_provider, notification, test_frontier_tracker(),
      iree_hal_pool_epoch_query_null(), allocator, &pool));

  iree_hal_pool_reservation_t first_reservation;
  iree_hal_pool_acquire_info_t reserve_info;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(
      AcquireOneReservation(pool, 13, 16, /*requester_frontier=*/NULL,
                            IREE_HAL_POOL_RESERVE_FLAG_NONE, &first_reservation,
                            &reserve_info, &result));
  ASSERT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);
  ReleaseOneReservation(pool, &first_reservation,
                        /*death_frontier=*/NULL);

  iree_hal_pool_reservation_t second_reservation;
  IREE_ASSERT_OK(
      AcquireOneReservation(pool, 13, 16, /*requester_frontier=*/NULL,
                            IREE_HAL_POOL_RESERVE_FLAG_NONE,
                            &second_reservation, &reserve_info, &result));
  ASSERT_TRUE(result == IREE_HAL_POOL_ACQUIRE_OK ||
              result == IREE_HAL_POOL_ACQUIRE_OK_FRESH);
  EXPECT_EQ(second_reservation.offset, first_reservation.offset);

  ReleaseOneReservation(pool, &second_reservation,
                        /*death_frontier=*/NULL);
  iree_hal_pool_release(pool);
  iree_async_notification_release(notification);
  iree_hal_slab_provider_release(slab_provider);
}

TEST_F(TLSFPoolTest, QueryCapabilitiesAndBudget) {
  iree_hal_pool_capabilities_t capabilities;
  iree_hal_pool_query_capabilities(pool_, &capabilities);
  EXPECT_TRUE(iree_all_bits_set(capabilities.memory_type,
                                IREE_HAL_MEMORY_TYPE_HOST_LOCAL));
  EXPECT_TRUE(iree_all_bits_set(capabilities.supported_usage,
                                IREE_HAL_BUFFER_USAGE_TRANSFER));
  EXPECT_EQ(capabilities.min_allocation_size, 1u);
  EXPECT_EQ(capabilities.max_allocation_size, 0u);
  EXPECT_GE(capabilities.max_allocation_alignment, 4096u);

  iree_hal_pool_release(pool_);
  pool_ = NULL;

  iree_hal_tlsf_pool_options_t options = DefaultOptions();
  options.budget_limit = 63;
  IREE_ASSERT_OK(CreateTLSFPool(
      options, slab_provider_, notification_, test_frontier_tracker(),
      iree_hal_pool_epoch_query_null(), allocator_, &pool_));

  iree_hal_pool_reservation_t reservation;
  iree_hal_pool_acquire_info_t reserve_info;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(AcquireOneReservation(
      pool_, 64, 16, /*requester_frontier=*/NULL,
      IREE_HAL_POOL_RESERVE_FLAG_NONE, &reservation, &reserve_info, &result));
  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_OVER_BUDGET);
}

}  // namespace
