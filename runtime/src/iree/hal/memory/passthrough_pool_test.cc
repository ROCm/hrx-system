// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/memory/passthrough_pool.h"

#include <atomic>
#include <thread>

#include "iree/async/frontier_tracker.h"
#include "iree/async/notification.h"
#include "iree/async/proactor.h"
#include "iree/async/proactor_platform.h"
#include "iree/hal/api.h"
#include "iree/hal/memory/cpu_slab_provider.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

static iree_async_frontier_tracker_t* test_frontier_tracker() {
  static iree_async_frontier_tracker_t* tracker = nullptr;
  if (!tracker) {
    IREE_CHECK_OK(iree_async_frontier_tracker_create(
        iree_async_frontier_tracker_options_default(), iree_allocator_system(),
        &tracker));
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

typedef struct iree_hal_test_opaque_slab_provider_t {
  // Base slab provider interface.
  iree_hal_slab_provider_t base;

  // Test-controlled native completion, independent of the pool implementation.
  struct {
    // Retain the next native acquire descriptor until explicitly completed.
    bool acquire;
    // Retain native release descriptors until explicitly completed.
    bool release;
    // Borrowed acquire descriptor; only one acquisition is delayed at a time.
    iree_hal_slab_acquire_request_t* acquisition;
    // Intrusive native return list using acquisition-secured descriptor
    // storage.
    iree_hal_slab_operation_t* releases;
  } deferred;

  // Optional external witness that the last provider reference was released.
  uint32_t* destruction_count;

  // Number of trim calls received by the provider.
  uint32_t trim_count;

  // Retention policy from the most recent trim request.
  iree_hal_pool_trim_flags_t last_trim_flags;

  // Number of wrap_buffer calls received by the provider.
  iree_atomic_int32_t wrap_count;

  // Number of native slabs freed by release_slab.
  iree_atomic_int32_t release_count;

  // One-based wrap call index that returns an injected error, or zero.
  int32_t fail_wrap_at;

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
                                    host_allocator, &provider->base);
  *out_provider = &provider->base;
  return iree_ok_status();
}

static void iree_hal_test_opaque_slab_provider_destroy(
    iree_hal_slab_provider_t* base_provider) {
  iree_hal_test_opaque_slab_provider_t* provider =
      (iree_hal_test_opaque_slab_provider_t*)base_provider;
  if (provider->destruction_count) {
    ++*provider->destruction_count;
  }
  iree_allocator_free(provider->base.host_allocator, provider);
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
      provider->base.host_allocator, min_length, IREE_HAL_HEAP_BUFFER_ALIGNMENT,
      /*offset=*/0, &backing));
  out_slab->base_ptr = (uint8_t*)(uintptr_t)1;
  out_slab->length = min_length;
  out_slab->provider_handle = (uint64_t)(uintptr_t)backing;
  out_slab->native_allocation.byte_length = min_length;
  out_slab->native_allocation.granularity = 1;
  out_slab->maintenance_range = iree_hal_make_buffer_range(0, min_length);
  out_slab->host_cacheability = IREE_HAL_HOST_CACHEABILITY_WRITE_BACK;
  out_slab->allocation_alignment = IREE_HAL_HEAP_BUFFER_ALIGNMENT;
  return iree_ok_status();
}

static void iree_hal_test_opaque_slab_provider_release_slab(
    iree_hal_slab_provider_t* base_provider, const iree_hal_slab_t* slab) {
  iree_hal_test_opaque_slab_provider_t* provider =
      (iree_hal_test_opaque_slab_provider_t*)base_provider;
  iree_allocator_free_aligned(provider->base.host_allocator,
                              (void*)(uintptr_t)slab->provider_handle);
  iree_atomic_fetch_add(&provider->release_count, 1, iree_memory_order_relaxed);
}

static iree_status_t iree_hal_test_opaque_slab_provider_wrap_buffer(
    iree_hal_slab_provider_t* base_provider, const iree_hal_slab_t* slab,
    iree_device_size_t slab_offset, iree_device_size_t allocation_size,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage,
    iree_hal_buffer_release_callback_t release_callback,
    iree_hal_buffer_t** out_buffer) {
  iree_hal_test_opaque_slab_provider_t* provider =
      (iree_hal_test_opaque_slab_provider_t*)base_provider;
  const int32_t wrap_index = iree_atomic_fetch_add(&provider->wrap_count, 1,
                                                   iree_memory_order_relaxed) +
                             1;
  if (wrap_index == provider->fail_wrap_at) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "injected wrap failure at call %d", wrap_index);
  }
  iree_byte_span_t data = iree_make_byte_span(
      (uint8_t*)(uintptr_t)slab->provider_handle + slab_offset,
      (iree_host_size_t)allocation_size);
  return iree_hal_heap_buffer_wrap(
      iree_hal_buffer_placement_undefined(),
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE |
          IREE_HAL_MEMORY_TYPE_HOST_COHERENT | IREE_HAL_MEMORY_TYPE_HOST_CACHED,
      allowed_access, allowed_usage, allocation_size, data, release_callback,
      provider->base.host_allocator, out_buffer);
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

static void iree_hal_test_opaque_slab_provider_acquire_slabs(
    iree_hal_slab_provider_t* provider,
    iree_hal_slab_acquire_request_t* request) {
  auto* native =
      reinterpret_cast<iree_hal_test_opaque_slab_provider_t*>(provider);
  if (native->deferred.acquire) {
    ASSERT_EQ(native->deferred.acquisition, nullptr);
    native->deferred.acquisition = request;
    return;
  }
  iree_hal_slab_provider_acquire_slabs_inline(
      provider, request, iree_hal_test_opaque_slab_provider_acquire_slab,
      iree_hal_test_opaque_slab_provider_release_slab);
}

static void iree_hal_test_opaque_slab_provider_release_slabs(
    iree_hal_slab_provider_t* provider,
    iree_hal_slab_release_request_t* request) {
  auto* native =
      reinterpret_cast<iree_hal_test_opaque_slab_provider_t*>(provider);
  if (native->deferred.release) {
    request->operation.next = native->deferred.releases;
    native->deferred.releases = &request->operation;
    return;
  }
  iree_hal_slab_provider_release_slabs_inline(
      provider, request, iree_hal_test_opaque_slab_provider_release_slab);
}

static void CompleteNativeAcquire(
    iree_hal_test_opaque_slab_provider_t* provider) {
  auto* request = provider->deferred.acquisition;
  ASSERT_NE(request, nullptr);
  provider->deferred.acquisition = nullptr;
  iree_hal_slab_provider_acquire_slabs_inline(
      &provider->base, request, iree_hal_test_opaque_slab_provider_acquire_slab,
      iree_hal_test_opaque_slab_provider_release_slab);
}

static void CompleteNativeRelease(
    iree_hal_test_opaque_slab_provider_t* provider) {
  auto* operation = provider->deferred.releases;
  ASSERT_NE(operation, nullptr);
  provider->deferred.releases = operation->next;
  iree_hal_slab_provider_release_slabs_inline(
      &provider->base,
      reinterpret_cast<iree_hal_slab_release_request_t*>(operation),
      iree_hal_test_opaque_slab_provider_release_slab);
}

const iree_hal_slab_provider_vtable_t
    iree_hal_test_opaque_slab_provider_vtable = {
        .destroy = iree_hal_test_opaque_slab_provider_destroy,
        .acquire_slabs = iree_hal_test_opaque_slab_provider_acquire_slabs,
        .release_slabs = iree_hal_test_opaque_slab_provider_release_slabs,
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

class PassthroughPoolTest : public ::testing::Test {
 protected:
  void SetUp() override {
    allocator_ = iree_allocator_system();
    IREE_ASSERT_OK(iree_hal_cpu_slab_provider_create(
        /*min_alignment=*/0, allocator_, &slab_provider_));
    IREE_ASSERT_OK(iree_async_notification_create(
        test_proactor(), IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification_));
    IREE_ASSERT_OK(iree_async_frontier_tracker_create(
        iree_async_frontier_tracker_options_default(), allocator_, &tracker_));
    iree_hal_passthrough_pool_options_t options = {};
    IREE_ASSERT_OK(iree_hal_passthrough_pool_create(
        options, slab_provider_, notification_, tracker_, allocator_, &pool_));
  }

  void TearDown() override {
    iree_hal_pool_release(pool_);
    iree_async_frontier_tracker_release(tracker_);
    iree_async_notification_release(notification_);
    iree_hal_slab_provider_release(slab_provider_);
  }

  iree_allocator_t allocator_;
  iree_hal_slab_provider_t* slab_provider_ = nullptr;
  iree_async_notification_t* notification_ = nullptr;
  iree_hal_pool_t* pool_ = nullptr;
  // Independent coordinate space for each test, including terminal failures.
  iree_async_frontier_tracker_t* tracker_ = nullptr;
};

static iree_hal_pool_reservation_request_t MakeReservationRequest(
    iree_device_size_t allocation_size, iree_device_size_t min_alignment) {
  return iree_hal_make_pool_reservation_request(allocation_size, min_alignment);
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
  return iree_hal_pool_acquire_reservations_sync(
      pool, 1, &request, requester_frontier, out_reservation, out_info,
      out_result, flags);
}

static void ReleaseOneReservation(
    iree_hal_pool_t* pool, const iree_hal_pool_reservation_t* reservation,
    const iree_async_frontier_t* death_frontier) {
  iree_hal_pool_release_reservations(pool, 1, reservation, death_frontier);
}

static iree_status_t MaterializeOneReservation(
    iree_hal_pool_t* pool, const iree_hal_pool_reservation_t* reservation,
    iree_hal_pool_materialize_flags_t flags, iree_hal_buffer_t** out_buffer) {
  const iree_hal_pool_reservation_request_t request =
      iree_hal_make_pool_reservation_request(reservation->byte_length, 1);
  return iree_hal_pool_materialize_reservations(pool, 1, &request, reservation,
                                                flags, out_buffer);
}

static iree_hal_asan_pool_options_t ShadowOptions() {
  iree_hal_asan_pool_options_t options = {
      .mode = IREE_HAL_ASAN_POOL_MODE_SHADOW,
      .shadow_granule_size = 8,
      .redzone_size = 16,
      .backing_alignment = IREE_HAL_HEAP_BUFFER_ALIGNMENT,
  };
  return options;
}

TEST_F(PassthroughPoolTest, ReserveRelease) {
  iree_hal_pool_reservation_t reservation;
  iree_hal_pool_acquire_info_t reserve_info;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(AcquireOneReservation(pool_, 4096, 1, NULL,
                                       IREE_HAL_POOL_RESERVE_FLAG_NONE,
                                       &reservation, &reserve_info, &result));
  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);
  EXPECT_EQ(reservation.offset, 0u);
  EXPECT_EQ(reservation.byte_length, 4096u);
  EXPECT_NE(reservation.block_handle, 0u);
  EXPECT_EQ(reserve_info.reuse_frontier, nullptr);
  EXPECT_EQ(reserve_info.flags, IREE_HAL_POOL_ACQUIRE_FLAG_NONE);

  ReleaseOneReservation(pool_, &reservation, NULL);
}

TEST_F(PassthroughPoolTest, AlignmentStrengthensNativeBackingGuarantee) {
  auto request = MakeReservationRequest(256, 4096);
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool_, request, iree_infinite_timeout(), &buffer));
  EXPECT_EQ(iree_hal_buffer_memory_view(buffer).backing->allocation_alignment,
            4096u);
  iree_hal_buffer_mapping_t mapping = {};
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      buffer, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_ALL,
      IREE_HAL_BUFFER_MAP_FLAG_NONE, 0, 256, &mapping));
  EXPECT_EQ(reinterpret_cast<uintptr_t>(mapping.contents.data) % 4096, 0u);
  memset(mapping.contents.data, 0x6B, 256);
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));
  uint8_t actual[256] = {};
  IREE_ASSERT_OK(iree_hal_buffer_map_read(buffer, 0, actual, sizeof(actual)));
  for (uint8_t value : actual) {
    EXPECT_EQ(value, 0x6B);
  }
  iree_hal_buffer_release(buffer);
}

// Counts real CPU backing and pool metadata allocations without replacing the
// pool or provider under test.
struct CountingAllocator {
  // Calls that can allocate storage.
  size_t allocation_count = 0;
  // Native and metadata allocations still owned.
  size_t live_count = 0;
  // Reject the selected allocating call; zero permits all calls.
  size_t fail_at = 0;
  // The public owner has returned all of this allocator's storage.
  bool closed = false;
  // Calls made after that owner's lifetime ended.
  size_t calls_after_close = 0;
  // Thread performing the last native allocation; frees do not change it.
  std::thread::id last_allocation_thread;

  static iree_status_t Control(void* self, iree_allocator_command_t command,
                               const void* params, void** inout_pointer) {
    auto* state = static_cast<CountingAllocator*>(self);
    if (state->closed) {
      ++state->calls_after_close;
    }
    if (command != IREE_ALLOCATOR_COMMAND_FREE) {
      ++state->allocation_count;
      state->last_allocation_thread = std::this_thread::get_id();
      if (state->allocation_count == state->fail_at) {
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "allocation unavailable");
      }
    }
    const bool had_pointer = *inout_pointer != nullptr;
    iree_allocator_t allocator = iree_allocator_system();
    iree_status_t status =
        allocator.ctl(allocator.self, command, params, inout_pointer);
    if (iree_status_is_ok(status)) {
      if (command == IREE_ALLOCATOR_COMMAND_FREE && had_pointer) {
        --state->live_count;
      } else if (command != IREE_ALLOCATOR_COMMAND_FREE && !had_pointer) {
        ++state->live_count;
      }
    }
    return status;
  }

  iree_allocator_t allocator() { return {this, Control}; }
};

class NativePassthroughPoolTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_hal_test_opaque_slab_provider_create(
        allocations_.allocator(), &provider_));
    native_ =
        reinterpret_cast<iree_hal_test_opaque_slab_provider_t*>(provider_);
    native_->destruction_count = &destruction_count_;
    IREE_ASSERT_OK(iree_async_notification_create(
        test_proactor(), IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification_));
    IREE_ASSERT_OK(iree_async_frontier_tracker_create(
        iree_async_frontier_tracker_options_default(), iree_allocator_system(),
        &tracker_));
    IREE_ASSERT_OK(
        iree_hal_passthrough_pool_create({}, provider_, notification_, tracker_,
                                         allocations_.allocator(), &pool_));
  }
  void TearDown() override {
    iree_hal_pool_release(pool_);
    iree_hal_slab_provider_release(provider_);
    iree_async_frontier_tracker_release(tracker_);
    iree_async_notification_release(notification_);
    EXPECT_EQ(destruction_count_, 1u);
    EXPECT_EQ(allocations_.live_count, 0u);
  }

  // Real allocations owned by the pool and its native dependency.
  CountingAllocator allocations_;
  // External destruction witness remains live after the provider is freed.
  uint32_t destruction_count_ = 0;
  // Owned native provider reference.
  iree_hal_slab_provider_t* provider_ = nullptr;
  // Borrowed native dependency whose completion is explicitly controlled.
  iree_hal_test_opaque_slab_provider_t* native_ = nullptr;
  // Owned capacity notification.
  iree_async_notification_t* notification_ = nullptr;
  // Owned independent release-history tracker.
  iree_async_frontier_tracker_t* tracker_ = nullptr;
  // Owned public source pool reference.
  iree_hal_pool_t* pool_ = nullptr;
};

TEST_F(NativePassthroughPoolTest,
       DeferredBatchPublishesTogetherAndReleaseSurvivesPoolDestruction) {
  native_->deferred.acquire = true;
  native_->deferred.release = true;
  const iree_hal_pool_reservation_request_t requests[] = {
      MakeReservationRequest(256, 16), MakeReservationRequest(2048, 16)};
  iree_hal_pool_reservation_t reservations[2];
  memset(reservations, 0xA5, sizeof(reservations));
  iree_hal_pool_reservation_t original[2];
  memcpy(original, reservations, sizeof(original));
  iree_hal_pool_acquire_info_t infos[2];
  memset(infos, 0x5A, sizeof(infos));
  iree_hal_pool_acquire_info_t original_infos[2];
  memcpy(original_infos, infos, sizeof(infos));
  int completions = 0;
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations(
      pool_, 2, requests, nullptr, reservations, infos,
      {[](void* user_data, iree_status_t status,
          iree_hal_pool_acquire_result_t result) {
         IREE_EXPECT_OK(status);
         EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);
         ++*static_cast<int*>(user_data);
       },
       &completions},
      IREE_HAL_POOL_RESERVE_FLAG_NONE));
  EXPECT_EQ(completions, 0);
  EXPECT_EQ(memcmp(original, reservations, sizeof(original)), 0);
  EXPECT_EQ(memcmp(original_infos, infos, sizeof(infos)), 0);
  ASSERT_NE(native_->deferred.acquisition, nullptr);
  EXPECT_EQ(native_->deferred.acquisition->count, 2u);
  CompleteNativeAcquire(native_);
  ASSERT_EQ(completions, 1);
  for (size_t i = 0; i < 2; ++i) {
    iree_hal_buffer_t* buffer = nullptr;
    IREE_ASSERT_OK(MaterializeOneReservation(
        pool_, &reservations[i], IREE_HAL_POOL_MATERIALIZE_FLAG_NONE, &buffer));
    const uint8_t pattern = static_cast<uint8_t>(0xC0 + i);
    IREE_ASSERT_OK(iree_hal_buffer_map_fill(buffer, 0, IREE_HAL_WHOLE_BUFFER,
                                            &pattern, sizeof(pattern)));
    uint8_t actual = 0;
    IREE_ASSERT_OK(iree_hal_buffer_map_read(
        buffer, reservations[i].byte_length - 1, &actual, sizeof(actual)));
    EXPECT_EQ(actual, pattern);
    iree_hal_buffer_release(buffer);
  }
  const size_t allocated = allocations_.allocation_count;
  allocations_.fail_at = allocated + 1;
  iree_hal_pool_release_reservations(pool_, 2, reservations, nullptr);
  EXPECT_EQ(allocations_.allocation_count, allocated);
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool_, &stats);
  EXPECT_EQ(stats.reservation_count, 0u);
  EXPECT_EQ(stats.bytes_reserved, 0u);
  EXPECT_EQ(stats.bytes_committed, 2304u);
  EXPECT_EQ(stats.slab_count, 2u);
  iree_hal_pool_release(pool_);
  pool_ = nullptr;
  iree_hal_slab_provider_release(provider_);
  provider_ = nullptr;
  EXPECT_EQ(destruction_count_, 0u);
  CompleteNativeRelease(native_);
  EXPECT_EQ(destruction_count_, 0u);
  CompleteNativeRelease(native_);
  native_ = nullptr;
  EXPECT_EQ(destruction_count_, 1u);
  EXPECT_EQ(allocations_.allocation_count, allocated);
}

TEST_F(NativePassthroughPoolTest,
       NativeRetirementDoesNotBorrowThePublicPoolAllocator) {
  iree_hal_pool_release(pool_);
  pool_ = nullptr;
  CountingAllocator public_allocator;
  IREE_ASSERT_OK(
      iree_hal_passthrough_pool_create({}, provider_, notification_, tracker_,
                                       public_allocator.allocator(), &pool_));
  native_->deferred.release = true;
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(
      iree_hal_pool_allocate_buffer(pool_, MakeReservationRequest(256, 16),
                                    iree_infinite_timeout(), &buffer));
  const uint32_t pattern = 0xCAFE1234;
  IREE_ASSERT_OK(
      iree_hal_buffer_map_fill(buffer, 0, 256, &pattern, sizeof(pattern)));
  uint32_t result = 0;
  IREE_ASSERT_OK(
      iree_hal_buffer_map_read(buffer, 252, &result, sizeof(result)));
  EXPECT_EQ(result, pattern);
  iree_hal_buffer_release(buffer);
  ASSERT_NE(native_->deferred.releases, nullptr);
  iree_hal_pool_release(pool_);
  pool_ = nullptr;
  EXPECT_EQ(public_allocator.live_count, 0u);
  public_allocator.closed = true;
  const size_t allocation_count = allocations_.allocation_count;
  CompleteNativeRelease(native_);
  EXPECT_EQ(public_allocator.calls_after_close, 0u);
  EXPECT_EQ(allocations_.allocation_count, allocation_count);
}

TEST_F(NativePassthroughPoolTest,
       NativeMetadataFailureReturnsThePublicPoolAllocation) {
  CountingAllocator public_allocator;
  allocations_.fail_at = allocations_.allocation_count + 1;
  iree_hal_pool_t* pool = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_hal_passthrough_pool_create({}, provider_, notification_, tracker_,
                                       public_allocator.allocator(), &pool));
  EXPECT_EQ(pool, nullptr);
  EXPECT_EQ(public_allocator.live_count, 0u);
}

TEST_F(NativePassthroughPoolTest, CompletionMayReenterAndReleaseThePool) {
  struct Call {
    // Owned public pool reference, released by terminal completion.
    iree_hal_pool_t* pool;
    // Borrowed acquisition geometry for the outer request.
    iree_hal_pool_reservation_request_t request;
    // Outer token, published before the callback.
    iree_hal_pool_reservation_t reservation;
    // Outer acquisition facts.
    iree_hal_pool_acquire_info_t info;
    // Exactly-once completion witness.
    int completions;
  } call = {pool_, MakeReservationRequest(256, 16), {}, {}, 0};
  native_->deferred.acquire = true;
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations(
      call.pool, 1, &call.request, nullptr, &call.reservation, &call.info,
      {[](void* user_data, iree_status_t status,
          iree_hal_pool_acquire_result_t result) {
         auto* call = static_cast<Call*>(user_data);
         IREE_EXPECT_OK(status);
         EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);
         EXPECT_NE(call->reservation.block_handle, 0u);
         iree_hal_pool_reservation_t nested;
         iree_hal_pool_acquire_info_t info;
         IREE_ASSERT_OK(iree_hal_pool_acquire_reservations_sync(
             call->pool, 1, &call->request, nullptr, &nested, &info, &result,
             IREE_HAL_POOL_RESERVE_FLAG_NONE));
         ReleaseOneReservation(call->pool, &nested, nullptr);
         ReleaseOneReservation(call->pool, &call->reservation, nullptr);
         ++call->completions;
         iree_hal_pool_release(call->pool);
         call->pool = nullptr;
       },
       &call},
      IREE_HAL_POOL_RESERVE_FLAG_NONE));
  native_->deferred.acquire = false;
  CompleteNativeAcquire(native_);
  EXPECT_EQ(call.completions, 1);
  EXPECT_EQ(call.pool, nullptr);
  pool_ = nullptr;
}

TEST_F(NativePassthroughPoolTest, NativeFailureRollsBackBeforeCompletion) {
  native_->deferred.acquire = true;
  const size_t baseline_live = allocations_.live_count;
  const iree_hal_pool_reservation_request_t requests[] = {
      MakeReservationRequest(256, 16), MakeReservationRequest(2048, 16)};
  iree_hal_pool_reservation_t reservations[2];
  memset(reservations, 0xA5, sizeof(reservations));
  iree_hal_pool_reservation_t original[2];
  memcpy(original, reservations, sizeof(original));
  iree_hal_pool_acquire_info_t infos[2];
  memset(infos, 0x5A, sizeof(infos));
  iree_hal_pool_acquire_info_t original_infos[2];
  memcpy(original_infos, infos, sizeof(infos));
  int completions = 0;
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations(
      pool_, 2, requests, nullptr, reservations, infos,
      {[](void* user_data, iree_status_t status,
          iree_hal_pool_acquire_result_t result) {
         IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, status);
         EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_NONE);
         ++*static_cast<int*>(user_data);
       },
       &completions},
      IREE_HAL_POOL_RESERVE_FLAG_NONE));
  // Metadata is already secured. Reject the second native allocation.
  allocations_.fail_at = allocations_.allocation_count + 2;
  CompleteNativeAcquire(native_);
  EXPECT_EQ(completions, 1);
  EXPECT_EQ(
      iree_atomic_load(&native_->release_count, iree_memory_order_relaxed), 1);
  EXPECT_EQ(memcmp(original, reservations, sizeof(original)), 0);
  EXPECT_EQ(memcmp(original_infos, infos, sizeof(infos)), 0);
  EXPECT_EQ(allocations_.live_count, baseline_live);
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool_, &stats);
  EXPECT_EQ(stats.reservation_count, 0u);
  EXPECT_EQ(stats.bytes_committed, 0u);
  EXPECT_EQ(stats.slab_count, 0u);
}

TEST_F(NativePassthroughPoolTest, CaptureFailureNeverStartsNativeAcquisition) {
  native_->deferred.acquire = true;
  const size_t baseline_live = allocations_.live_count;
  const iree_hal_pool_reservation_request_t requests[] = {
      MakeReservationRequest(256, 16), MakeReservationRequest(2048, 16)};
  for (size_t failure_offset : {1u, 2u, 3u}) {
    allocations_.fail_at = allocations_.allocation_count + failure_offset;
    iree_hal_pool_reservation_t reservations[2] = {};
    iree_hal_pool_acquire_info_t infos[2] = {};
    int completions = 0;
    IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                          iree_hal_pool_acquire_reservations(
                              pool_, 2, requests, nullptr, reservations, infos,
                              {[](void* user_data, iree_status_t status,
                                  iree_hal_pool_acquire_result_t result) {
                                 IREE_EXPECT_OK(status);
                                 ++*static_cast<int*>(user_data);
                               },
                               &completions},
                              IREE_HAL_POOL_RESERVE_FLAG_NONE));
    EXPECT_EQ(completions, 0);
    EXPECT_EQ(native_->deferred.acquisition, nullptr);
    EXPECT_EQ(allocations_.live_count, baseline_live);
    EXPECT_EQ(reservations[0].block_handle, 0u);
    EXPECT_EQ(reservations[1].block_handle, 0u);
  }
}

TEST_F(NativePassthroughPoolTest,
       RetainedViewAndFrontierBothPrecedeNativeReturn) {
  native_->deferred.release = true;
  const auto request = MakeReservationRequest(256, 16);
  iree_hal_pool_reservation_t reservation;
  iree_hal_pool_acquire_info_t info;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations_sync(
      pool_, 1, &request, nullptr, &reservation, &info, &result,
      IREE_HAL_POOL_RESERVE_FLAG_NONE));
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(MaterializeOneReservation(
      pool_, &reservation, IREE_HAL_POOL_MATERIALIZE_FLAG_NONE, &buffer));
  const iree_async_axis_t axis = iree_async_axis_make_queue(1, 0, 0, 0, 0);
  IREE_ASSERT_OK(
      iree_async_frontier_tracker_register_axis(tracker_, axis, nullptr));
  iree_async_single_frontier_t frontier = {.entry_count = 1};
  frontier.entries[0] = {axis, 1};
  ReleaseOneReservation(pool_, &reservation,
                        iree_async_fixed_frontier_as_const_frontier(&frontier));
  EXPECT_EQ(native_->deferred.releases, nullptr);
  iree_async_frontier_tracker_advance(tracker_, axis, 1);
  EXPECT_EQ(native_->deferred.releases, nullptr);
  iree_hal_buffer_release(buffer);
  ASSERT_NE(native_->deferred.releases, nullptr);
  CompleteNativeRelease(native_);
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool_, &stats);
  EXPECT_EQ(stats.bytes_committed, 0u);
}

TEST_F(NativePassthroughPoolTest,
       DestructionCancelsWaitWithoutJoiningNativeFree) {
  native_->deferred.release = true;
  iree_hal_pool_reservation_t reservation;
  iree_hal_pool_acquire_info_t info;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(AcquireOneReservation(pool_, 256, 16, nullptr,
                                       IREE_HAL_POOL_RESERVE_FLAG_NONE,
                                       &reservation, &info, &result));
  const iree_async_axis_t axis = iree_async_axis_make_queue(1, 0, 0, 0, 0);
  IREE_ASSERT_OK(
      iree_async_frontier_tracker_register_axis(tracker_, axis, nullptr));
  iree_async_single_frontier_t frontier = {.entry_count = 1};
  frontier.entries[0] = {axis, 1};
  ReleaseOneReservation(pool_, &reservation,
                        iree_async_fixed_frontier_as_const_frontier(&frontier));
  EXPECT_EQ(native_->deferred.releases, nullptr);
  iree_hal_pool_release(pool_);
  pool_ = nullptr;
  ASSERT_NE(native_->deferred.releases, nullptr);
  EXPECT_EQ(iree_async_frontier_tracker_advance(tracker_, axis, 1), 0u);
  iree_async_frontier_tracker_release(tracker_);
  tracker_ = nullptr;
  iree_hal_slab_provider_release(provider_);
  provider_ = nullptr;
  EXPECT_EQ(destruction_count_, 0u);
  CompleteNativeRelease(native_);
  native_ = nullptr;
  EXPECT_EQ(destruction_count_, 1u);
}

TEST(PassthroughPool, ScalarAndBatchAcquisitionCompleteOnInlineProvider) {
  CountingAllocator allocations;
  iree_hal_slab_provider_t* provider = nullptr;
  IREE_ASSERT_OK(iree_hal_cpu_slab_provider_create(
      /*min_alignment=*/0, allocations.allocator(), &provider));
  iree_async_notification_t* notification = nullptr;
  IREE_ASSERT_OK(iree_async_notification_create(
      test_proactor(), IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification));
  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(iree_hal_passthrough_pool_create(
      {}, provider, notification, test_frontier_tracker(),
      allocations.allocator(), &pool));

  // Exercise scalar and batched native geometry through the same interface.
  for (iree_host_size_t request_count : {1u, 9u}) {
    iree_hal_pool_reservation_request_t requests[9];
    iree_hal_pool_reservation_t reservations[9];
    iree_hal_pool_acquire_info_t infos[9];
    memset(infos, 0x5A, sizeof(infos));
    for (iree_host_size_t i = 0; i < request_count; ++i) {
      requests[i] = MakeReservationRequest(128 + i * 16, 16);
    }
    const size_t allocation_count = allocations.allocation_count;
    iree_hal_pool_acquire_result_t result = IREE_HAL_POOL_ACQUIRE_NONE;
    // The inline provider allocates directly and produces usable buffers.
    IREE_ASSERT_OK(iree_hal_pool_acquire_reservations_sync(
        pool, request_count, requests,
        /*requester_frontier=*/nullptr, reservations, infos, &result,
        IREE_HAL_POOL_RESERVE_FLAG_ALLOW_WAIT_FRONTIER));
    ASSERT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);
    EXPECT_GT(allocations.allocation_count, allocation_count);
    EXPECT_EQ(allocations.last_allocation_thread, std::this_thread::get_id());
    iree_hal_buffer_t* buffers[9];
    IREE_ASSERT_OK(iree_hal_pool_materialize_reservations(
        pool, request_count, requests, reservations,
        IREE_HAL_POOL_MATERIALIZE_FLAG_TRANSFER_RESERVATION_OWNERSHIP,
        buffers));
    for (iree_host_size_t i = 0; i < request_count; ++i) {
      const uint8_t pattern = static_cast<uint8_t>(i + 1);
      IREE_ASSERT_OK(iree_hal_buffer_map_fill(
          buffers[i], 0, IREE_HAL_WHOLE_BUFFER, &pattern, sizeof(pattern)));
      uint8_t actual = 0;
      IREE_ASSERT_OK(iree_hal_buffer_map_read(buffers[i],
                                              requests[i].allocation_size - 1,
                                              &actual, sizeof(actual)));
      EXPECT_EQ(actual, pattern);
      iree_hal_buffer_release(buffers[i]);
    }
  }
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool, &stats);
  EXPECT_EQ(stats.exhausted_count, 0u);
  EXPECT_EQ(stats.reservation_count, 0u);
  EXPECT_EQ(stats.bytes_committed, 0u);
  iree_hal_pool_release(pool);
  iree_async_notification_release(notification);
  iree_hal_slab_provider_release(provider);
}

TEST_F(PassthroughPoolTest, ReservationTransactionValidatesBeforeAcquiring) {
  const iree_hal_pool_reservation_request_t requests[2] = {
      MakeReservationRequest(1024, 16),
      MakeReservationRequest(1024, 3),
  };
  iree_hal_pool_reservation_t reservations[2];
  memset(reservations, 0xA5, sizeof(reservations));
  iree_hal_pool_reservation_t original_reservations[2];
  memcpy(original_reservations, reservations, sizeof(original_reservations));
  iree_hal_pool_acquire_info_t infos[2];
  memset(infos, 0x5A, sizeof(infos));
  iree_hal_pool_acquire_info_t original_infos[2];
  memcpy(original_infos, infos, sizeof(original_infos));
  iree_hal_pool_acquire_result_t result = IREE_HAL_POOL_ACQUIRE_OVER_BUDGET;
  for (iree_hal_pool_reserve_flags_t flags :
       {IREE_HAL_POOL_RESERVE_FLAG_NONE,
        IREE_HAL_POOL_RESERVE_FLAG_ALLOW_WAIT_FRONTIER}) {
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        iree_hal_pool_acquire_reservations_sync(
            pool_, IREE_ARRAYSIZE(requests), requests,
            /*requester_frontier=*/NULL, reservations, infos, &result, flags));
  }

  EXPECT_EQ(memcmp(reservations, original_reservations, sizeof(reservations)),
            0);
  EXPECT_EQ(memcmp(infos, original_infos, sizeof(infos)), 0);
  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_OVER_BUDGET);
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool_, &stats);
  EXPECT_EQ(stats.reservation_count, 0u);
  EXPECT_EQ(stats.bytes_reserved, 0u);
  EXPECT_EQ(stats.slab_count, 0u);
  EXPECT_EQ(stats.reserve_count, 0u);
}

TEST_F(PassthroughPoolTest,
       MaterializationTransactionTransfersAllReservations) {
  const iree_hal_pool_reservation_request_t requests[3] = {
      MakeReservationRequest(64, 16),
      MakeReservationRequest(128, 16),
      MakeReservationRequest(256, 16),
  };
  iree_hal_pool_reservation_t reservations[3];
  iree_hal_pool_acquire_info_t infos[3];
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations_sync(
      pool_, IREE_ARRAYSIZE(requests), requests,
      /*requester_frontier=*/NULL, reservations, infos, &result,
      IREE_HAL_POOL_RESERVE_FLAG_NONE));
  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);

  iree_hal_buffer_t* buffers[3];
  IREE_ASSERT_OK(iree_hal_pool_materialize_reservations(
      pool_, IREE_ARRAYSIZE(reservations), requests, reservations,
      IREE_HAL_POOL_MATERIALIZE_FLAG_TRANSFER_RESERVATION_OWNERSHIP, buffers));

  iree_hal_buffer_release(buffers[0]);
  iree_hal_buffer_release(buffers[1]);
  iree_hal_buffer_release(buffers[2]);
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool_, &stats);
  EXPECT_EQ(stats.reservation_count, 0u);
  EXPECT_EQ(stats.slab_count, 0u);
  EXPECT_EQ(stats.reserve_count, 3u);
  EXPECT_EQ(stats.release_count, 3u);
}

TEST(PassthroughPool, FailedMaterializationTransactionRetainsEveryReservation) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_slab_provider_t* slab_provider = NULL;
  IREE_ASSERT_OK(
      iree_hal_test_opaque_slab_provider_create(allocator, &slab_provider));
  auto* test_provider = (iree_hal_test_opaque_slab_provider_t*)slab_provider;
  test_provider->fail_wrap_at = 2;
  iree_async_notification_t* notification = NULL;
  IREE_ASSERT_OK(iree_async_notification_create(
      test_proactor(), IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification));
  iree_hal_passthrough_pool_options_t options = {};
  iree_hal_pool_t* pool = NULL;
  IREE_ASSERT_OK(iree_hal_passthrough_pool_create(
      options, slab_provider, notification, test_frontier_tracker(), allocator,
      &pool));

  const iree_hal_pool_reservation_request_t requests[3] = {
      MakeReservationRequest(64, 16),
      MakeReservationRequest(128, 16),
      MakeReservationRequest(256, 16),
  };
  iree_hal_pool_reservation_t reservations[3];
  iree_hal_pool_acquire_info_t infos[3];
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations_sync(
      pool, IREE_ARRAYSIZE(requests), requests,
      /*requester_frontier=*/NULL, reservations, infos, &result,
      IREE_HAL_POOL_RESERVE_FLAG_NONE));

  iree_hal_buffer_t* buffers[3];
  memset(buffers, 0xA5, sizeof(buffers));
  iree_hal_buffer_t* original_buffers[3];
  memcpy(original_buffers, buffers, sizeof(original_buffers));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INTERNAL,
      iree_hal_pool_materialize_reservations(
          pool, IREE_ARRAYSIZE(reservations), requests, reservations,
          IREE_HAL_POOL_MATERIALIZE_FLAG_TRANSFER_RESERVATION_OWNERSHIP,
          buffers));
  EXPECT_EQ(memcmp(buffers, original_buffers, sizeof(buffers)), 0);

  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool, &stats);
  EXPECT_EQ(stats.reservation_count, 3u);
  EXPECT_EQ(stats.release_count, 0u);

  iree_hal_pool_release_reservations(pool, IREE_ARRAYSIZE(reservations),
                                     reservations,
                                     /*death_frontier=*/NULL);
  iree_hal_pool_query_stats(pool, &stats);
  EXPECT_EQ(stats.reservation_count, 0u);
  EXPECT_EQ(stats.release_count, 3u);

  iree_hal_pool_release(pool);
  iree_async_notification_release(notification);
  iree_hal_slab_provider_release(slab_provider);
}

TEST_F(PassthroughPoolTest, ReserveRejectsUnsupportedAlignment) {
  // This native provider has a fixed alignment limit. The ordinary host
  // provider can satisfy larger alignments separately on each acquisition.
  iree_hal_slab_provider_t* provider = nullptr;
  IREE_ASSERT_OK(
      iree_hal_test_opaque_slab_provider_create(allocator_, &provider));
  iree_hal_pool_release(pool_);
  IREE_ASSERT_OK(iree_hal_passthrough_pool_create(
      {}, provider, notification_, tracker_, allocator_, &pool_));
  iree_hal_slab_provider_release(provider);
  iree_hal_pool_reservation_t reservation;
  iree_hal_pool_acquire_info_t reserve_info;
  iree_hal_pool_acquire_result_t result;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      AcquireOneReservation(pool_, 4096, IREE_HAL_HEAP_BUFFER_ALIGNMENT * 2,
                            NULL, IREE_HAL_POOL_RESERVE_FLAG_NONE, &reservation,
                            &reserve_info, &result));
}

TEST_F(PassthroughPoolTest, StatsTrackReserveRelease) {
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool_, &stats);
  EXPECT_EQ(stats.reservation_count, 0u);
  EXPECT_EQ(stats.bytes_reserved, 0u);
  EXPECT_EQ(stats.reserve_count, 0u);
  EXPECT_EQ(stats.release_count, 0u);

  iree_hal_pool_reservation_t reservation;
  iree_hal_pool_acquire_info_t reserve_info;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(AcquireOneReservation(pool_, 1024, 1, NULL,
                                       IREE_HAL_POOL_RESERVE_FLAG_NONE,
                                       &reservation, &reserve_info, &result));

  iree_hal_pool_query_stats(pool_, &stats);
  EXPECT_EQ(stats.reservation_count, 1u);
  EXPECT_GE(stats.bytes_reserved, 1024u);
  EXPECT_EQ(stats.reserve_count, 1u);
  EXPECT_EQ(stats.release_count, 0u);
  EXPECT_EQ(stats.fresh_count, 1u);

  ReleaseOneReservation(pool_, &reservation, NULL);

  iree_hal_pool_query_stats(pool_, &stats);
  EXPECT_EQ(stats.reservation_count, 0u);
  EXPECT_EQ(stats.bytes_reserved, 0u);
  EXPECT_EQ(stats.reserve_count, 1u);
  EXPECT_EQ(stats.release_count, 1u);
}

TEST_F(PassthroughPoolTest, WrapReservationCreatesBuffer) {
  iree_hal_pool_reservation_t reservation;
  iree_hal_pool_acquire_info_t reserve_info;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(AcquireOneReservation(pool_, 4096, 1, NULL,
                                       IREE_HAL_POOL_RESERVE_FLAG_NONE,
                                       &reservation, &reserve_info, &result));

  iree_hal_buffer_t* buffer = NULL;
  IREE_ASSERT_OK(MaterializeOneReservation(
      pool_, &reservation,
      IREE_HAL_POOL_MATERIALIZE_FLAG_TRANSFER_RESERVATION_OWNERSHIP, &buffer));
  ASSERT_NE(buffer, nullptr);
  EXPECT_EQ(iree_hal_buffer_allocation_size(buffer), 4096u);
  EXPECT_EQ(iree_hal_buffer_byte_length(buffer), 4096u);

  // Releasing the buffer should release the reservation back to the pool.
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool_, &stats);
  EXPECT_EQ(stats.reservation_count, 1u);

  iree_hal_buffer_release(buffer);

  iree_hal_pool_query_stats(pool_, &stats);
  EXPECT_EQ(stats.reservation_count, 0u);
  EXPECT_EQ(stats.release_count, 1u);
}

TEST_F(PassthroughPoolTest, BorrowedMaterializationKeepsSlabUntilViewRelease) {
  iree_hal_pool_reservation_t reservation;
  iree_hal_pool_acquire_info_t acquire_info;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(AcquireOneReservation(
      pool_, 1024, 1, /*requester_frontier=*/NULL,
      IREE_HAL_POOL_RESERVE_FLAG_NONE, &reservation, &acquire_info, &result));

  iree_hal_buffer_t* buffer = NULL;
  IREE_ASSERT_OK(MaterializeOneReservation(
      pool_, &reservation, IREE_HAL_POOL_MATERIALIZE_FLAG_NONE, &buffer));

  ReleaseOneReservation(pool_, &reservation,
                        /*death_frontier=*/NULL);

  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool_, &stats);
  EXPECT_EQ(stats.reservation_count, 0u);
  EXPECT_EQ(stats.release_count, 1u);
  EXPECT_EQ(stats.slab_count, 1u);

  iree_hal_buffer_release(buffer);

  iree_hal_pool_query_stats(pool_, &stats);
  EXPECT_EQ(stats.slab_count, 0u);
}

TEST_F(PassthroughPoolTest,
       RetirementCopiesEveryAxisRegisteredAfterAllocation) {
  iree_hal_pool_reservation_t reservation;
  iree_hal_pool_acquire_info_t info;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(AcquireOneReservation(pool_, 4096, 16, nullptr,
                                       IREE_HAL_POOL_RESERVE_FLAG_NONE,
                                       &reservation, &info, &result));

  // Even a sealed device group permits new queue identities. The allocation
  // precedes registration and must still accept the complete release history.
  IREE_ASYNC_FIXED_FRONTIER_TYPE(FullFrontier, UINT8_MAX);
  FullFrontier frontier = {.entry_count = UINT8_MAX};
  for (uint32_t i = 0; i < frontier.entry_count; ++i) {
    frontier.entries[i] = {
        iree_async_axis_make_queue(1, 0, 0, static_cast<uint8_t>(i), 0), i + 1};
    IREE_ASSERT_OK(iree_async_frontier_tracker_register_axis(
        tracker_, frontier.entries[i].axis, nullptr));
  }
  ReleaseOneReservation(pool_, &reservation,
                        iree_async_fixed_frontier_as_const_frontier(&frontier));
  // The caller can immediately reuse its frontier storage.
  memset(&frontier, 0, sizeof(frontier));
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool_, &stats);
  EXPECT_EQ(stats.reservation_count, 0u);
  EXPECT_EQ(stats.bytes_reserved, 0u);
  EXPECT_EQ(stats.bytes_committed, 4096u);

  // All but the final axis have completed. A trim cannot turn a partial
  // completion into permission to free native storage.
  for (uint32_t i = 0; i + 1 < UINT8_MAX; ++i) {
    EXPECT_EQ(
        iree_async_frontier_tracker_advance(
            tracker_,
            iree_async_axis_make_queue(1, 0, 0, static_cast<uint8_t>(i), 0),
            i + 1),
        0u);
  }
  iree_hal_pool_trim(pool_, IREE_HAL_POOL_TRIM_FLAG_ALL, 0);
  iree_hal_pool_query_stats(pool_, &stats);
  EXPECT_EQ(stats.bytes_committed, 4096u);
  EXPECT_EQ(iree_async_frontier_tracker_advance(
                tracker_, iree_async_axis_make_queue(1, 0, 0, UINT8_MAX - 1, 0),
                UINT8_MAX),
            1u);
  iree_hal_pool_query_stats(pool_, &stats);
  EXPECT_EQ(stats.bytes_committed, 0u);
  EXPECT_EQ(stats.slab_count, 0u);
}

TEST_F(PassthroughPoolTest, ReturnedTokenAndViewsBothPrecedeRetirement) {
  auto request = MakeReservationRequest(256, 16);
  iree_hal_pool_reservation_t reservation;
  iree_hal_pool_acquire_info_t info;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations_sync(
      pool_, 1, &request, nullptr, &reservation, &info, &result,
      IREE_HAL_POOL_RESERVE_FLAG_NONE));
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(MaterializeOneReservation(
      pool_, &reservation, IREE_HAL_POOL_MATERIALIZE_FLAG_NONE, &buffer));
  const uint32_t pattern = 0xABCDEF10u;
  IREE_ASSERT_OK(
      iree_hal_buffer_map_fill(buffer, 0, 256, &pattern, sizeof(pattern)));
  uint32_t actual = 0;
  IREE_ASSERT_OK(
      iree_hal_buffer_map_read(buffer, 252, &actual, sizeof(actual)));
  EXPECT_EQ(actual, pattern);
  const iree_async_axis_t axis = iree_async_axis_make_queue(1, 0, 0, 0, 0);
  IREE_ASSERT_OK(
      iree_async_frontier_tracker_register_axis(tracker_, axis, nullptr));
  iree_async_single_frontier_t frontier = {.entry_count = 1};
  frontier.entries[0] = {axis, 1};
  ReleaseOneReservation(pool_, &reservation,
                        iree_async_fixed_frontier_as_const_frontier(&frontier));
  iree_async_frontier_tracker_advance(tracker_, axis, 1);
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool_, &stats);
  EXPECT_EQ(stats.bytes_committed, 256u);
  // Retaining the borrowed view postpones native cleanup, but does not grant
  // access after the caller has explicitly returned the allocation epoch.
  iree_hal_buffer_release(buffer);
  iree_hal_pool_query_stats(pool_, &stats);
  EXPECT_EQ(stats.bytes_committed, 0u);
}

TEST_F(PassthroughPoolTest, FailedAxisQuarantinesPendingNativeBacking) {
  iree_hal_pool_reservation_t reservation;
  iree_hal_pool_acquire_info_t info;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(AcquireOneReservation(pool_, 4096, 16, nullptr,
                                       IREE_HAL_POOL_RESERVE_FLAG_NONE,
                                       &reservation, &info, &result));
  IREE_ASYNC_FIXED_FRONTIER_TYPE(PairFrontier, 2);
  PairFrontier frontier = {.entry_count = 2};
  for (uint8_t i = 0; i < 2; ++i) {
    frontier.entries[i] = {iree_async_axis_make_queue(1, 0, 0, i, 0), 1};
    IREE_ASSERT_OK(iree_async_frontier_tracker_register_axis(
        tracker_, frontier.entries[i].axis, nullptr));
  }
  ReleaseOneReservation(pool_, &reservation,
                        iree_async_fixed_frontier_as_const_frontier(&frontier));
  const uint32_t observed =
      iree_async_notification_begin_observe(notification_);
  iree_async_frontier_tracker_fail_axis(
      tracker_, frontier.entries[0].axis,
      iree_make_status(IREE_STATUS_UNAVAILABLE, "device execution failed"));
  EXPECT_NE(iree_async_notification_query_epoch(notification_), observed);
  iree_async_notification_end_observe(notification_);
  iree_hal_pool_trim(pool_, IREE_HAL_POOL_TRIM_FLAG_ALL, 0);
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool_, &stats);
  EXPECT_EQ(stats.bytes_committed, 4096u);
  EXPECT_EQ(stats.reservation_count, 0u);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                        AcquireOneReservation(pool_, 16, 16, nullptr,
                                              IREE_HAL_POOL_RESERVE_FLAG_NONE,
                                              &reservation, &info, &result));

  // The caller retires the remaining device before final destruction. Failure
  // itself never certifies that another axis has stopped touching the bytes.
  iree_async_frontier_tracker_advance(tracker_, frontier.entries[1].axis, 1);
  iree_hal_pool_release(pool_);
  pool_ = nullptr;
}

TEST_F(PassthroughPoolTest,
       CallerQuiescentDestructionCancelsPendingRetirement) {
  iree_hal_pool_reservation_t reservation;
  iree_hal_pool_acquire_info_t info;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(AcquireOneReservation(pool_, 4096, 16, nullptr,
                                       IREE_HAL_POOL_RESERVE_FLAG_NONE,
                                       &reservation, &info, &result));
  iree_async_single_frontier_t frontier = {.entry_count = 1};
  frontier.entries[0] = {iree_async_axis_make_queue(1, 0, 0, 0, 0), 1};
  IREE_ASSERT_OK(iree_async_frontier_tracker_register_axis(
      tracker_, frontier.entries[0].axis, nullptr));
  ReleaseOneReservation(pool_, &reservation,
                        iree_async_fixed_frontier_as_const_frontier(&frontier));
  // No execution uses the allocation. Final pool release cancels its waiter,
  // so a later tracker update has no callback into retired pool metadata.
  iree_hal_pool_release(pool_);
  pool_ = nullptr;
  EXPECT_EQ(iree_async_frontier_tracker_advance(tracker_,
                                                frontier.entries[0].axis, 1),
            0u);
}

TEST_F(PassthroughPoolTest, TrimDoesNotJoinPendingRetirement) {
  iree_hal_slab_provider_t* slab_provider = nullptr;
  IREE_ASSERT_OK(
      iree_hal_test_opaque_slab_provider_create(allocator_, &slab_provider));
  auto* provider =
      reinterpret_cast<iree_hal_test_opaque_slab_provider_t*>(slab_provider);
  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(iree_hal_passthrough_pool_create(
      {}, slab_provider, notification_, tracker_, allocator_, &pool));
  iree_async_single_frontier_t frontier = {.entry_count = 1};
  frontier.entries[0] = {iree_async_axis_make_queue(1, 0, 0, 0, 0), 1};
  IREE_ASSERT_OK(iree_async_frontier_tracker_register_axis(
      tracker_, frontier.entries[0].axis, nullptr));

  const auto request = MakeReservationRequest(256, 16);
  iree_hal_pool_reservation_t reservation;
  iree_hal_pool_acquire_info_t info;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations_sync(
      pool, 1, &request, nullptr, &reservation, &info, &result,
      IREE_HAL_POOL_RESERVE_FLAG_NONE));
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(MaterializeOneReservation(
      pool, &reservation, IREE_HAL_POOL_MATERIALIZE_FLAG_NONE, &buffer));
  ReleaseOneReservation(pool, &reservation,
                        iree_async_fixed_frontier_as_const_frontier(&frontier));
  iree_hal_buffer_release(buffer);

  // Execution and caller views have retired, but frontier publication has
  // not. Trimming cannot infer physical completion.
  iree_hal_pool_trim(pool, IREE_HAL_POOL_TRIM_FLAG_ALL, 0);
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool, &stats);
  EXPECT_EQ(stats.reservation_count, 0u);
  EXPECT_EQ(stats.bytes_reserved, 0u);
  EXPECT_EQ(stats.bytes_committed, 256u);
  EXPECT_EQ(stats.slab_count, 1u);
  EXPECT_EQ(
      iree_atomic_load(&provider->release_count, iree_memory_order_relaxed), 0);

  // Caller-quiescent final release cancels the waiter and returns native
  // storage through the inline provider.
  iree_hal_pool_release(pool);
  EXPECT_EQ(
      iree_atomic_load(&provider->release_count, iree_memory_order_relaxed), 1);
  EXPECT_EQ(iree_async_frontier_tracker_advance(tracker_,
                                                frontier.entries[0].axis, 1),
            0u);
  iree_hal_slab_provider_release(slab_provider);
}

TEST_F(PassthroughPoolTest, FrontierDispatchRacesFinalPoolRelease) {
  const iree_async_axis_t axis = iree_async_axis_make_queue(1, 0, 0, 0, 0);
  IREE_ASSERT_OK(
      iree_async_frontier_tracker_register_axis(tracker_, axis, nullptr));
  // The bytes have no remaining device users. Race final caller teardown with
  // publication of their completion: cancellation may win, or the dispatched
  // callback retains private pool state through native release.
  for (uint64_t epoch = 1; epoch <= 64; ++epoch) {
    iree_hal_pool_reservation_t reservation;
    iree_hal_pool_acquire_info_t info;
    iree_hal_pool_acquire_result_t result;
    IREE_ASSERT_OK(AcquireOneReservation(pool_, 4096, 16, nullptr,
                                         IREE_HAL_POOL_RESERVE_FLAG_NONE,
                                         &reservation, &info, &result));
    iree_async_single_frontier_t frontier = {.entry_count = 1};
    frontier.entries[0] = {axis, epoch};
    ReleaseOneReservation(
        pool_, &reservation,
        iree_async_fixed_frontier_as_const_frontier(&frontier));
    std::atomic<bool> start{false};
    std::thread completer([&] {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      iree_async_frontier_tracker_advance(tracker_, axis, epoch);
    });
    start.store(true, std::memory_order_release);
    iree_hal_pool_release(pool_);
    pool_ = nullptr;
    completer.join();
    IREE_ASSERT_OK(iree_hal_passthrough_pool_create(
        {}, slab_provider_, notification_, tracker_, allocator_, &pool_));
  }
}

TEST_F(PassthroughPoolTest, FailedFrontierDispatchRacesFinalPoolRelease) {
  // Failure can publish quarantine while public destruction claims cleanup.
  // The callback must retain the notification owner through its final access.
  for (uint8_t ordinal = 1; ordinal <= 64; ++ordinal) {
    const iree_async_axis_t axis =
        iree_async_axis_make_queue(1, 0, 0, ordinal, 0);
    IREE_ASSERT_OK(
        iree_async_frontier_tracker_register_axis(tracker_, axis, nullptr));
    iree_hal_pool_reservation_t reservation;
    iree_hal_pool_acquire_info_t info;
    iree_hal_pool_acquire_result_t result;
    IREE_ASSERT_OK(AcquireOneReservation(pool_, 4096, 16, nullptr,
                                         IREE_HAL_POOL_RESERVE_FLAG_NONE,
                                         &reservation, &info, &result));
    iree_async_single_frontier_t frontier = {.entry_count = 1};
    frontier.entries[0] = {axis, 1};
    ReleaseOneReservation(
        pool_, &reservation,
        iree_async_fixed_frontier_as_const_frontier(&frontier));
    std::atomic<bool> start{false};
    std::thread completer([&] {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      iree_async_frontier_tracker_fail_axis(
          tracker_, axis,
          iree_make_status(IREE_STATUS_UNAVAILABLE, "device execution failed"));
    });
    start.store(true, std::memory_order_release);
    iree_hal_pool_release(pool_);
    pool_ = nullptr;
    completer.join();
    IREE_ASSERT_OK(iree_hal_passthrough_pool_create(
        {}, slab_provider_, notification_, tracker_, allocator_, &pool_));
  }
}

TEST(PassthroughPool, UsesProviderHooks) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_slab_provider_t* slab_provider = NULL;
  IREE_ASSERT_OK(
      iree_hal_test_opaque_slab_provider_create(allocator, &slab_provider));
  iree_async_notification_t* notification = NULL;
  IREE_ASSERT_OK(iree_async_notification_create(
      test_proactor(), IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification));

  iree_hal_pool_t* pool = NULL;
  iree_hal_passthrough_pool_options_t options = {};
  IREE_ASSERT_OK(iree_hal_passthrough_pool_create(
      options, slab_provider, notification, test_frontier_tracker(), allocator,
      &pool));

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
      pool, 256, 1, /*requester_frontier=*/NULL,
      IREE_HAL_POOL_RESERVE_FLAG_NONE, &reservation, &reserve_info, &result));

  iree_hal_buffer_t* buffer = NULL;
  IREE_ASSERT_OK(MaterializeOneReservation(
      pool, &reservation,
      IREE_HAL_POOL_MATERIALIZE_FLAG_TRANSFER_RESERVATION_OWNERSHIP, &buffer));

  // All trim modes preserve this live buffer and reach its native source.
  auto* provider = (iree_hal_test_opaque_slab_provider_t*)slab_provider;
  const iree_hal_pool_trim_flags_t trim_flags[] = {
      IREE_HAL_POOL_TRIM_FLAG_NONE,
      IREE_HAL_POOL_TRIM_FLAG_EXCESS,
      IREE_HAL_POOL_TRIM_FLAG_ALL,
      IREE_HAL_POOL_TRIM_FLAG_ALL | IREE_HAL_POOL_TRIM_FLAG_EXCESS,
  };
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(trim_flags); ++i) {
    iree_hal_pool_trim(pool, trim_flags[i], /*min_bytes_to_keep=*/0);
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
      IREE_HAL_BUFFER_MAP_FLAG_NONE, 0, 256, &mapping));
  memset(mapping.contents.data, 0x6B, 256);
  EXPECT_EQ(((uint8_t*)mapping.contents.data)[255], 0x6B);
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));

  EXPECT_EQ(
      1,
      iree_atomic_load(
          &((iree_hal_test_opaque_slab_provider_t*)slab_provider)->wrap_count,
          iree_memory_order_relaxed));

  iree_hal_buffer_release(buffer);
  iree_hal_pool_release(pool);
  iree_async_notification_release(notification);
  iree_hal_slab_provider_release(slab_provider);
}

TEST(PassthroughPool, CreateRejectsASANWhenProviderCannotAdviseRanges) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_slab_provider_t* slab_provider = NULL;
  IREE_ASSERT_OK(iree_hal_cpu_slab_provider_create(/*min_alignment=*/0,
                                                   allocator, &slab_provider));
  iree_async_notification_t* notification = NULL;
  IREE_ASSERT_OK(iree_async_notification_create(
      test_proactor(), IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification));

  iree_hal_passthrough_pool_options_t options = {.asan = ShadowOptions()};

  iree_hal_pool_t* pool = NULL;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_hal_passthrough_pool_create(
                            options, slab_provider, notification,
                            test_frontier_tracker(), allocator, &pool));
  EXPECT_EQ(pool, nullptr);

  iree_async_notification_release(notification);
  iree_hal_slab_provider_release(slab_provider);
}

TEST(PassthroughPool, ASANAdvisesBackingRangeAndExposesUserRange) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_slab_provider_t* slab_provider = NULL;
  IREE_ASSERT_OK(
      iree_hal_test_opaque_slab_provider_create(allocator, &slab_provider));
  iree_async_notification_t* notification = NULL;
  IREE_ASSERT_OK(iree_async_notification_create(
      test_proactor(), IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification));

  iree_hal_passthrough_pool_options_t options = {.asan = ShadowOptions()};

  iree_hal_pool_t* pool = NULL;
  IREE_ASSERT_OK(iree_hal_passthrough_pool_create(
      options, slab_provider, notification, test_frontier_tracker(), allocator,
      &pool));

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

  iree_hal_buffer_t* buffer = NULL;
  IREE_ASSERT_OK(MaterializeOneReservation(
      pool, &reservation,
      IREE_HAL_POOL_MATERIALIZE_FLAG_TRANSFER_RESERVATION_OWNERSHIP, &buffer));
  ASSERT_NE(buffer, nullptr);
  EXPECT_EQ(iree_hal_buffer_allocation_size(buffer), 13u);
  EXPECT_EQ(iree_hal_buffer_byte_length(buffer), 13u);

  iree_hal_buffer_release(buffer);

  // A completed allocation epoch is poisoned before logical release returns,
  // independently of whether the native worker has reclaimed its backing.
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
  ReleaseOneReservation(pool, &reservation, nullptr);
  EXPECT_EQ(2, iree_atomic_load(&provider->asan_released_count,
                                iree_memory_order_relaxed));

  // A reservation whose lifetime never starts returns without shadow effects,
  // even when its physical retirement carries an outstanding prerequisite.
  IREE_ASSERT_OK(AcquireOneReservation(pool, 13, 16, nullptr,
                                       IREE_HAL_POOL_RESERVE_FLAG_NONE,
                                       &reservation, &reserve_info, &result));
  ASSERT_EQ(IREE_HAL_POOL_ACQUIRE_OK_FRESH, result);
  const iree_async_axis_t axis = iree_async_axis_make_queue(1, 0, 1, 0, 0);
  IREE_ASSERT_OK(iree_async_frontier_tracker_register_axis(
      test_frontier_tracker(), axis, nullptr));
  iree_async_single_frontier_t frontier = {.entry_count = 1};
  frontier.entries[0] = {axis, 1};
  ReleaseOneReservation(pool, &reservation,
                        iree_async_fixed_frontier_as_const_frontier(&frontier));
  EXPECT_EQ(iree_atomic_load(&provider->asan_released_count,
                             iree_memory_order_relaxed),
            2);
  iree_async_frontier_tracker_advance(test_frontier_tracker(), axis, 1);
  EXPECT_EQ(iree_atomic_load(&provider->asan_released_count,
                             iree_memory_order_relaxed),
            2);
  EXPECT_LT(provider->last_asan_allocated_sequence,
            provider->last_asan_released_sequence);

  iree_hal_pool_release(pool);
  iree_async_notification_release(notification);
  iree_hal_slab_provider_release(slab_provider);
}

TEST_F(PassthroughPoolTest, BufferMemoryAccess) {
  iree_hal_pool_reservation_t reservation;
  iree_hal_pool_acquire_info_t reserve_info;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(AcquireOneReservation(pool_, 256, 1, NULL,
                                       IREE_HAL_POOL_RESERVE_FLAG_NONE,
                                       &reservation, &reserve_info, &result));

  iree_hal_buffer_t* buffer = NULL;
  IREE_ASSERT_OK(MaterializeOneReservation(
      pool_, &reservation,
      IREE_HAL_POOL_MATERIALIZE_FLAG_TRANSFER_RESERVATION_OWNERSHIP, &buffer));

  // Map, write, read back.
  iree_hal_buffer_mapping_t mapping;
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      buffer, IREE_HAL_MAPPING_MODE_SCOPED,
      IREE_HAL_MEMORY_ACCESS_WRITE | IREE_HAL_MEMORY_ACCESS_READ,
      IREE_HAL_BUFFER_MAP_FLAG_NONE, 0, 256, &mapping));
  memset(mapping.contents.data, 0xCD, 256);
  EXPECT_EQ(((uint8_t*)mapping.contents.data)[0], 0xCD);
  EXPECT_EQ(((uint8_t*)mapping.contents.data)[255], 0xCD);
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));

  iree_hal_buffer_release(buffer);
}

TEST_F(PassthroughPoolTest, Capabilities) {
  iree_hal_pool_capabilities_t capabilities;
  iree_hal_pool_query_capabilities(pool_, &capabilities);
  EXPECT_TRUE(iree_all_bits_set(capabilities.memory_type,
                                IREE_HAL_MEMORY_TYPE_HOST_LOCAL));
  EXPECT_TRUE(iree_all_bits_set(capabilities.supported_usage,
                                IREE_HAL_BUFFER_USAGE_TRANSFER));
  EXPECT_TRUE(iree_all_bits_set(capabilities.supported_usage,
                                IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED));
}

TEST_F(PassthroughPoolTest, TrimWithNoCachedBacking) {
  iree_hal_pool_trim(pool_, IREE_HAL_POOL_TRIM_FLAG_EXCESS,
                     /*min_bytes_to_keep=*/0);
}

TEST_F(PassthroughPoolTest, MultipleReservations) {
  iree_hal_pool_reservation_t reservations[4];
  iree_hal_pool_acquire_info_t reserve_info;
  iree_hal_pool_acquire_result_t result;
  for (int i = 0; i < 4; ++i) {
    IREE_ASSERT_OK(AcquireOneReservation(
        pool_, 1024 * (i + 1), 1, NULL, IREE_HAL_POOL_RESERVE_FLAG_NONE,
        &reservations[i], &reserve_info, &result));
    EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);
    EXPECT_EQ(reserve_info.reuse_frontier, nullptr);
    EXPECT_EQ(reserve_info.flags, IREE_HAL_POOL_ACQUIRE_FLAG_NONE);
  }

  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool_, &stats);
  EXPECT_EQ(stats.reservation_count, 4u);
  EXPECT_EQ(stats.slab_count, 4u);
  EXPECT_EQ(stats.reserve_count, 4u);

  for (int i = 3; i >= 0; --i) {
    ReleaseOneReservation(pool_, &reservations[i], NULL);
  }

  iree_hal_pool_query_stats(pool_, &stats);
  EXPECT_EQ(stats.reservation_count, 0u);
  EXPECT_EQ(stats.slab_count, 0u);
  EXPECT_EQ(stats.release_count, 4u);
}

TEST_F(PassthroughPoolTest, AllocateBuffer) {
  iree_hal_buffer_t* buffer = NULL;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool_, iree_hal_make_pool_reservation_request(2048, /*min_alignment=*/1),
      iree_make_timeout_ms(0), &buffer));
  ASSERT_NE(buffer, nullptr);
  EXPECT_EQ(iree_hal_buffer_allocation_size(buffer), 2048u);
  EXPECT_EQ(iree_hal_buffer_byte_length(buffer), 2048u);

  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool_, &stats);
  EXPECT_EQ(stats.reservation_count, 1u);

  iree_hal_buffer_release(buffer);

  iree_hal_pool_query_stats(pool_, &stats);
  EXPECT_EQ(stats.reservation_count, 0u);
}

TEST_F(PassthroughPoolTest, WrappedBuffersBorrowPool) {
  iree_hal_buffer_t* buffer = NULL;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool_, iree_hal_make_pool_reservation_request(512, /*min_alignment=*/1),
      iree_make_timeout_ms(0), &buffer));

  // Wrapped buffers borrow the pool. Use the buffer while the pool is still
  // alive, then release the buffer before releasing the pool.
  iree_hal_buffer_mapping_t mapping;
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      buffer, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_WRITE,
      IREE_HAL_BUFFER_MAP_FLAG_NONE, 0, 512, &mapping));
  memset(mapping.contents.data, 0xEF, 512);
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));

  iree_hal_buffer_release(buffer);
}

}  // namespace
