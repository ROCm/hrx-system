// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include "hrx_internal.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class ExactPoolTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_cpu_initialize(/*flags=*/0)));
    IREE_ASSERT_OK(
        hrx_status_to_iree(hrx_cpu_device_get(/*index=*/0, &device_)));
  }

  void TearDown() override {
    IREE_EXPECT_OK(hrx_status_to_iree(hrx_cpu_shutdown()));
  }

  iree_hal_buffer_params_t BufferParams() const {
    return {
        /*.usage=*/IREE_HAL_BUFFER_USAGE_DEFAULT |
            IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED,
        /*.access=*/IREE_HAL_MEMORY_ACCESS_ALL,
        /*.type=*/IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
            IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
    };
  }

  hrx_device_t device_ = nullptr;
};

TEST_F(ExactPoolTest, ReservationRetainsPoolUntilQueueRetirement) {
  const iree_hal_buffer_params_t params = BufferParams();
  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(hrx_iree_exact_pool_create(device_->allocator.hal_allocator,
                                            params, &pool));

  iree_hal_pool_reservation_t reservation;
  iree_hal_pool_acquire_info_t acquire_info;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservation(
      pool, /*size=*/4096, /*alignment=*/1, /*requester_frontier=*/nullptr,
      IREE_HAL_POOL_RESERVE_FLAG_NONE, &reservation, &acquire_info, &result));
  ASSERT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);

  iree_hal_buffer_t* borrowed_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_pool_materialize_reservation(
      pool, params, &reservation, IREE_HAL_POOL_MATERIALIZE_FLAG_NONE,
      &borrowed_buffer));

  // Model the HRX wrapper being released after execution completes while the
  // queue still owns the transient buffer and its reservation. The reservation
  // retain must keep |pool| alive through the later retirement callback.
  iree_hal_pool_release(pool);
  iree_hal_buffer_release(borrowed_buffer);
  iree_hal_pool_release_reservation(pool, &reservation,
                                    /*death_frontier=*/nullptr);
}

TEST_F(ExactPoolTest, TransferredReservationDropsPoolRetain) {
  const iree_hal_buffer_params_t params = BufferParams();
  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(hrx_iree_exact_pool_create(device_->allocator.hal_allocator,
                                            params, &pool));

  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool, params, /*allocation_size=*/4096, /*requester_frontier=*/nullptr,
      iree_infinite_timeout(), &buffer));

  iree_hal_buffer_release(buffer);
  iree_hal_pool_release(pool);
}

TEST_F(ExactPoolTest, RejectsInvalidOrStrongerReservationAlignment) {
  iree_hal_buffer_params_t params = BufferParams();
  params.min_alignment = 8;
  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(hrx_iree_exact_pool_create(device_->allocator.hal_allocator,
                                            params, &pool));

  iree_hal_pool_reservation_t reservation;
  iree_hal_pool_acquire_info_t acquire_info;
  iree_hal_pool_acquire_result_t result;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_pool_acquire_reservation(pool, /*size=*/4096, /*alignment=*/0,
                                        /*requester_frontier=*/nullptr,
                                        IREE_HAL_POOL_RESERVE_FLAG_NONE,
                                        &reservation, &acquire_info, &result));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_pool_acquire_reservation(pool, /*size=*/4096, /*alignment=*/3,
                                        /*requester_frontier=*/nullptr,
                                        IREE_HAL_POOL_RESERVE_FLAG_NONE,
                                        &reservation, &acquire_info, &result));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_pool_acquire_reservation(pool, /*size=*/4096, /*alignment=*/16,
                                        /*requester_frontier=*/nullptr,
                                        IREE_HAL_POOL_RESERVE_FLAG_NONE,
                                        &reservation, &acquire_info, &result));

  IREE_ASSERT_OK(iree_hal_pool_acquire_reservation(
      pool, /*size=*/4096, /*alignment=*/8,
      /*requester_frontier=*/nullptr, IREE_HAL_POOL_RESERVE_FLAG_NONE,
      &reservation, &acquire_info, &result));
  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);
  iree_hal_pool_release_reservation(pool, &reservation,
                                    /*death_frontier=*/nullptr);
  iree_hal_pool_release(pool);
}

}  // namespace
