// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/memory/passthrough_pool.h"

#include "iree/async/notification.h"
#include "iree/async/proactor.h"
#include "iree/async/proactor_platform.h"
#include "iree/hal/api.h"
#include "iree/hal/memory/cpu_slab_provider.h"
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

typedef struct iree_hal_test_opaque_slab_provider_t {
  // Base slab provider interface.
  iree_hal_slab_provider_t base;

  // Host allocator used for provider metadata and slab allocations.
  iree_allocator_t host_allocator;

  // Number of wrap_buffer calls received by the provider.
  iree_atomic_int32_t wrap_count;

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
    iree_hal_slab_t* out_slab) {
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
    iree_hal_slab_provider_t* base_provider, iree_hal_slab_t* slab) {}

static void iree_hal_test_opaque_slab_provider_trim(
    iree_hal_slab_provider_t* base_provider,
    iree_hal_slab_provider_trim_flags_t flags) {}

static void iree_hal_test_opaque_slab_provider_query_stats(
    const iree_hal_slab_provider_t* base_provider,
    iree_hal_slab_provider_visited_set_t* visited,
    iree_hal_slab_provider_stats_t* out_stats) {
  iree_hal_slab_provider_visited(visited, base_provider);
}

static void iree_hal_test_opaque_slab_provider_query_properties(
    const iree_hal_slab_provider_t* base_provider,
    iree_hal_slab_provider_properties_t* out_properties) {
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
        /*.destroy=*/iree_hal_test_opaque_slab_provider_destroy,
        /*.acquire_slab=*/iree_hal_test_opaque_slab_provider_acquire_slab,
        /*.release_slab=*/iree_hal_test_opaque_slab_provider_release_slab,
        /*.wrap_buffer=*/iree_hal_test_opaque_slab_provider_wrap_buffer,
        /*.validate_asan_options=*/
        iree_hal_test_opaque_slab_provider_validate_asan_options,
        /*.advise_asan_range=*/
        iree_hal_test_opaque_slab_provider_advise_asan_range,
        /*.prefault=*/iree_hal_test_opaque_slab_provider_prefault,
        /*.trim=*/iree_hal_test_opaque_slab_provider_trim,
        /*.query_stats=*/iree_hal_test_opaque_slab_provider_query_stats,
        /*.query_properties=*/
        iree_hal_test_opaque_slab_provider_query_properties,
};

class PassthroughPoolTest : public ::testing::Test {
 protected:
  void SetUp() override {
    allocator_ = iree_allocator_system();
    IREE_ASSERT_OK(
        iree_hal_cpu_slab_provider_create(allocator_, &slab_provider_));
    IREE_ASSERT_OK(iree_async_notification_create(
        test_proactor(), IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification_));
    iree_hal_passthrough_pool_options_t options = {};
    IREE_ASSERT_OK(iree_hal_passthrough_pool_create(
        options, slab_provider_, notification_, allocator_, &pool_));
  }

  void TearDown() override {
    iree_hal_pool_release(pool_);
    iree_async_notification_release(notification_);
    iree_hal_slab_provider_release(slab_provider_);
  }

  iree_allocator_t allocator_;
  iree_hal_slab_provider_t* slab_provider_ = nullptr;
  iree_async_notification_t* notification_ = nullptr;
  iree_hal_pool_t* pool_ = nullptr;
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

static iree_hal_asan_pool_options_t ShadowOptions() {
  iree_hal_asan_pool_options_t options = {};
  options.mode = IREE_HAL_ASAN_POOL_MODE_SHADOW;
  options.shadow_granule_size = 8;
  options.redzone_size = 16;
  options.backing_alignment = IREE_HAL_HEAP_BUFFER_ALIGNMENT;
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
  EXPECT_EQ(reserve_info.wait_frontier, nullptr);
  EXPECT_EQ(reserve_info.flags, IREE_HAL_POOL_ACQUIRE_FLAG_NONE);

  ReleaseOneReservation(pool_, &reservation, NULL);
}

TEST_F(PassthroughPoolTest, ReservationTransactionValidatesBeforeAcquiring) {
  const iree_hal_pool_reservation_request_t requests[2] = {
      MakeReservationRequest(1024, 16),
      MakeReservationRequest(1024, IREE_HAL_HEAP_BUFFER_ALIGNMENT * 2),
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
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_pool_acquire_reservations(
          pool_, IREE_ARRAYSIZE(requests), requests,
          /*requester_frontier=*/NULL, IREE_HAL_POOL_RESERVE_FLAG_NONE,
          reservations, infos, &result));

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
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations(
      pool_, IREE_ARRAYSIZE(requests), requests,
      /*requester_frontier=*/NULL, IREE_HAL_POOL_RESERVE_FLAG_NONE,
      reservations, infos, &result));
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
      options, slab_provider, notification, allocator, &pool));

  const iree_hal_pool_reservation_request_t requests[3] = {
      MakeReservationRequest(64, 16),
      MakeReservationRequest(128, 16),
      MakeReservationRequest(256, 16),
  };
  iree_hal_pool_reservation_t reservations[3];
  iree_hal_pool_acquire_info_t infos[3];
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations(
      pool, IREE_ARRAYSIZE(requests), requests,
      /*requester_frontier=*/NULL, IREE_HAL_POOL_RESERVE_FLAG_NONE,
      reservations, infos, &result));

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

  iree_hal_buffer_params_t params = {0};
  params.usage =
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
  params.type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;

  iree_hal_buffer_t* buffer = NULL;
  IREE_ASSERT_OK(MaterializeOneReservation(pool_, params, &reservation,
                                           IREE_HAL_POOL_MATERIALIZE_FLAG_NONE,
                                           &buffer));

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
      options, slab_provider, notification, allocator, &pool));

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

  iree_hal_buffer_params_t params = {0};
  params.usage =
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
  params.type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;

  iree_hal_buffer_t* buffer = NULL;
  IREE_ASSERT_OK(MaterializeOneReservation(
      pool, params, &reservation,
      IREE_HAL_POOL_MATERIALIZE_FLAG_TRANSFER_RESERVATION_OWNERSHIP, &buffer));

  iree_hal_buffer_mapping_t mapping;
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      buffer, IREE_HAL_MAPPING_MODE_SCOPED,
      IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE, 0, 256,
      &mapping));
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
  IREE_ASSERT_OK(iree_hal_cpu_slab_provider_create(allocator, &slab_provider));
  iree_async_notification_t* notification = NULL;
  IREE_ASSERT_OK(iree_async_notification_create(
      test_proactor(), IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification));

  iree_hal_passthrough_pool_options_t options = {};
  options.asan = ShadowOptions();

  iree_hal_pool_t* pool = NULL;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_hal_passthrough_pool_create(options, slab_provider, notification,
                                       allocator, &pool));
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

  iree_hal_passthrough_pool_options_t options = {};
  options.asan = ShadowOptions();

  iree_hal_pool_t* pool = NULL;
  IREE_ASSERT_OK(iree_hal_passthrough_pool_create(
      options, slab_provider, notification, allocator, &pool));

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
  EXPECT_EQ(iree_hal_buffer_allocation_size(buffer), 13u);
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
  EXPECT_EQ(iree_atomic_load(&provider->asan_allocated_count,
                             iree_memory_order_relaxed),
            2);
  EXPECT_LT(first_release_sequence, provider->last_asan_allocated_sequence);

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

TEST_F(PassthroughPoolTest, BufferMemoryAccess) {
  iree_hal_pool_reservation_t reservation;
  iree_hal_pool_acquire_info_t reserve_info;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(AcquireOneReservation(pool_, 256, 1, NULL,
                                       IREE_HAL_POOL_RESERVE_FLAG_NONE,
                                       &reservation, &reserve_info, &result));

  iree_hal_buffer_params_t params = {0};
  params.usage =
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
  params.type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;

  iree_hal_buffer_t* buffer = NULL;
  IREE_ASSERT_OK(MaterializeOneReservation(
      pool_, params, &reservation,
      IREE_HAL_POOL_MATERIALIZE_FLAG_TRANSFER_RESERVATION_OWNERSHIP, &buffer));

  // Map, write, read back.
  iree_hal_buffer_mapping_t mapping;
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      buffer, IREE_HAL_MAPPING_MODE_SCOPED,
      IREE_HAL_MEMORY_ACCESS_WRITE | IREE_HAL_MEMORY_ACCESS_READ, 0, 256,
      &mapping));
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

TEST_F(PassthroughPoolTest, TrimIsNoOp) {
  IREE_EXPECT_OK(iree_hal_pool_trim(pool_));
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
    EXPECT_EQ(reserve_info.wait_frontier, nullptr);
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
  iree_hal_buffer_params_t params = {0};
  params.usage =
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
  params.type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;

  iree_hal_buffer_t* buffer = NULL;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool_, params, 2048, NULL, iree_make_timeout_ms(0), &buffer));
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
  iree_hal_buffer_params_t params = {0};
  params.usage =
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
  params.type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;

  iree_hal_buffer_t* buffer = NULL;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool_, params, 512, NULL, iree_make_timeout_ms(0), &buffer));

  // Wrapped buffers borrow the pool. Use the buffer while the pool is still
  // alive, then release the buffer before releasing the pool.
  iree_hal_buffer_mapping_t mapping;
  IREE_ASSERT_OK(iree_hal_buffer_map_range(buffer, IREE_HAL_MAPPING_MODE_SCOPED,
                                           IREE_HAL_MEMORY_ACCESS_WRITE, 0, 512,
                                           &mapping));
  memset(mapping.contents.data, 0xEF, 512);
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));

  iree_hal_buffer_release(buffer);
}

}  // namespace
