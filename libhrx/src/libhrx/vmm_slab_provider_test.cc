// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include "vmm_slab_provider.h"

#include "hrx_internal.h"
#include "iree/hal/memory/tlsf_pool.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class VmmSlabProviderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree::Status status(hrx_status_to_iree(hrx_gpu_initialize(/*flags=*/0)));
    if (status.code() == iree::StatusCode::kUnavailable ||
        status.code() == iree::StatusCode::kNotFound) {
      GTEST_SKIP() << status.ToString();
    }
    IREE_ASSERT_OK(status);
    initialized_ = true;
    hrx_device_t device = nullptr;
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_gpu_device_get(0, &device)));
    if (!iree_hal_allocator_supports_virtual_memory(
            device->allocator.hal_allocator)) {
      GTEST_SKIP() << "The device allocator does not support virtual memory.";
    }
    params_.usage = IREE_HAL_BUFFER_USAGE_DEFAULT;
    params_.access = IREE_HAL_MEMORY_ACCESS_ALL;
    params_.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
    params_.queue_family_affinity = IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY;
    IREE_ASSERT_OK(
        hrx_vmm_slab_provider_create(device->allocator.hal_allocator, params_,
                                     iree_allocator_system(), &provider_));
    iree_hal_queue_pool_backend_t backend;
    IREE_ASSERT_OK(iree_hal_device_query_queue_pool_backend(
        device->hal_device, iree_hal_queue_family(device->transfer_queue),
        &backend));
    iree_hal_tlsf_pool_options_t options = {};
    options.tlsf_options.range_length = 2 * 1024 * 1024;
    options.tlsf_options.alignment = 256;
    options.tlsf_options.frontier_capacity =
        IREE_HAL_MEMORY_TLSF_DEFAULT_FRONTIER_CAPACITY;
    IREE_ASSERT_OK(iree_hal_tlsf_pool_create(
        options, provider_, backend.notification, backend.epoch_query,
        iree_allocator_system(), &pool_));
  }

  void TearDown() override {
    iree_hal_pool_release(pool_);
    iree_hal_slab_provider_release(provider_);
    if (initialized_) {
      IREE_EXPECT_OK(hrx_status_to_iree(hrx_gpu_shutdown()));
    }
  }

  iree_hal_pool_stats_t PoolStats() {
    iree_hal_pool_stats_t stats = {};
    iree_hal_pool_query_stats(pool_, &stats);
    return stats;
  }

  iree_hal_slab_provider_stats_t ProviderStats() {
    iree_hal_slab_provider_visited_set_t visited = {};
    iree_hal_slab_provider_stats_t stats = {};
    iree_hal_slab_provider_query_stats(provider_, &visited, &stats);
    return stats;
  }

  // Whether the fixture owns an initialized GPU runtime.
  bool initialized_ = false;
  // Device-local parameters matching the production VMM pool configuration.
  iree_hal_buffer_params_t params_ = {};
  // Production provider owning the native virtual reservation and mapping.
  iree_hal_slab_provider_t* provider_ = nullptr;
  // Real TLSF pool whose reservation and slab lifetimes are observed.
  iree_hal_pool_t* pool_ = nullptr;
};

TEST_F(VmmSlabProviderTest, OwnedRangeSurvivesUntilItsFinalSubspan) {
  iree_hal_buffer_t* owner = nullptr;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool_, params_, 4096, nullptr, iree_infinite_timeout(), &owner));
  iree_hal_buffer_t* root = iree_hal_buffer_allocated_buffer(owner);
  ASSERT_NE(owner, root);
  iree_hal_buffer_t* intermediate = nullptr;
  IREE_ASSERT_OK(iree_hal_buffer_subspan(
      owner, 256, 1024, iree_allocator_system(), &intermediate));
  iree_hal_buffer_t* child = nullptr;
  IREE_ASSERT_OK(iree_hal_buffer_subspan(intermediate, 64, 128,
                                         iree_allocator_system(), &child));
  iree_hal_buffer_t* sibling = nullptr;
  IREE_ASSERT_OK(iree_hal_buffer_subspan(owner, 1024, 128,
                                         iree_allocator_system(), &sibling));
  EXPECT_EQ(root, iree_hal_buffer_allocated_buffer(child));
  EXPECT_EQ(iree_hal_buffer_byte_offset(owner) + 320,
            iree_hal_buffer_byte_offset(child));
  iree_hal_buffer_release(owner);
  iree_hal_buffer_release(intermediate);

  EXPECT_EQ(1u, PoolStats().reservation_count);
  EXPECT_EQ(0u, PoolStats().release_count);
  IREE_ASSERT_OK(iree_hal_pool_trim(pool_));
  EXPECT_EQ(0u, ProviderStats().total_released);
  iree_hal_buffer_release(child);
  EXPECT_EQ(1u, PoolStats().reservation_count);
  iree_hal_buffer_release(sibling);
  EXPECT_EQ(0u, PoolStats().reservation_count);
  EXPECT_EQ(1u, PoolStats().release_count);

  IREE_ASSERT_OK(iree_hal_pool_trim(pool_));
  EXPECT_EQ(0u, PoolStats().slab_count);
  EXPECT_EQ(0u, PoolStats().bytes_committed);
  EXPECT_EQ(1u, ProviderStats().total_acquired);
  EXPECT_EQ(1u, ProviderStats().total_released);
}

TEST_F(VmmSlabProviderTest, BorrowedSubspanDoesNotOwnTheReservation) {
  const iree_hal_pool_reservation_request_t request = {params_, 4096};
  iree_hal_pool_reservation_t reservation;
  iree_hal_pool_acquire_info_t info;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations(
      pool_, 1, &request, nullptr, IREE_HAL_POOL_RESERVE_FLAG_NONE,
      &reservation, &info, &result));
  ASSERT_EQ(IREE_HAL_POOL_ACQUIRE_OK_FRESH, result);
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_pool_materialize_reservations(
      pool_, 1, &request, &reservation, IREE_HAL_POOL_MATERIALIZE_FLAG_NONE,
      &buffer));
  iree_hal_buffer_t* child = nullptr;
  IREE_ASSERT_OK(iree_hal_buffer_subspan(buffer, 256, 128,
                                         iree_allocator_system(), &child));
  iree_hal_buffer_release(buffer);
  EXPECT_EQ(1u, PoolStats().reservation_count);

  // There are no device uses to order. Explicit retirement ends the reservation
  // even while a borrowed view still retains the allocation representation.
  iree_hal_pool_release_reservations(pool_, 1, &reservation, nullptr);
  EXPECT_EQ(0u, PoolStats().reservation_count);
  EXPECT_EQ(1u, PoolStats().release_count);
  iree_hal_buffer_release(child);
  EXPECT_EQ(1u, PoolStats().release_count);
  IREE_ASSERT_OK(iree_hal_pool_trim(pool_));
  EXPECT_EQ(1u, ProviderStats().total_released);
}

}  // namespace
