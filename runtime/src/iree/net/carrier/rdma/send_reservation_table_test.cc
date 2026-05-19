// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/carrier/rdma/send_reservation_table.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

static void CountLeaseRelease(void* user_data, uint32_t buffer_index) {
  uint32_t* release_count = (uint32_t*)user_data;
  EXPECT_EQ(11u, buffer_index);
  *release_count += 1;
}

static iree_async_buffer_lease_t MakeLease(uint32_t* release_count) {
  iree_async_buffer_lease_t lease = {};
  lease.release = (iree_async_buffer_recycle_callback_t){
      .fn = CountLeaseRelease,
      .user_data = release_count,
  };
  lease.buffer_index = 11u;
  lease.span = iree_async_span_from_ptr(reinterpret_cast<void*>(0x1000), 128);
  return lease;
}

class SendReservationTableTest : public ::testing::Test {
 protected:
  void TearDown() override {
    iree_net_rdma_send_reservation_table_deinitialize(&table_);
  }

  iree_status_t Initialize(uint32_t capacity) {
    return iree_net_rdma_send_reservation_table_initialize(
        capacity, iree_allocator_system(), &table_);
  }

  iree_net_rdma_send_reservation_table_t table_ = {};
};

TEST_F(SendReservationTableTest, AcquireAndResolveTransfersLease) {
  IREE_ASSERT_OK(Initialize(2));

  uint32_t release_count = 0;
  iree_async_buffer_lease_t lease = MakeLease(&release_count);
  iree_net_carrier_send_handle_t handle = 0;
  IREE_ASSERT_OK(iree_net_rdma_send_reservation_table_acquire(
      &table_, &lease, /*byte_length=*/64, &handle));
  EXPECT_EQ(nullptr, lease.release.fn);
  EXPECT_NE(0u, handle);
  EXPECT_EQ(1u,
            iree_net_rdma_send_reservation_table_available_capacity(&table_));

  iree_net_rdma_send_reservation_t reservation;
  IREE_ASSERT_OK(iree_net_rdma_send_reservation_table_resolve(&table_, handle,
                                                              &reservation));
  EXPECT_EQ(64u, reservation.byte_length);
  EXPECT_NE(nullptr, reservation.buffer_lease.release.fn);
  EXPECT_EQ(2u,
            iree_net_rdma_send_reservation_table_available_capacity(&table_));

  iree_async_buffer_lease_release(&reservation.buffer_lease);
  iree_async_buffer_lease_release(&reservation.buffer_lease);
  EXPECT_EQ(1u, release_count);
}

TEST_F(SendReservationTableTest, AbortReleasesLease) {
  IREE_ASSERT_OK(Initialize(1));

  uint32_t release_count = 0;
  iree_async_buffer_lease_t lease = MakeLease(&release_count);
  iree_net_carrier_send_handle_t handle = 0;
  IREE_ASSERT_OK(iree_net_rdma_send_reservation_table_acquire(
      &table_, &lease, /*byte_length=*/32, &handle));

  IREE_ASSERT_OK(iree_net_rdma_send_reservation_table_abort(&table_, handle));
  EXPECT_EQ(1u, release_count);
  EXPECT_EQ(1u,
            iree_net_rdma_send_reservation_table_available_capacity(&table_));
}

TEST_F(SendReservationTableTest, DeinitializeReleasesOutstandingLeases) {
  IREE_ASSERT_OK(Initialize(1));

  uint32_t release_count = 0;
  iree_async_buffer_lease_t lease = MakeLease(&release_count);
  iree_net_carrier_send_handle_t handle = 0;
  IREE_ASSERT_OK(iree_net_rdma_send_reservation_table_acquire(
      &table_, &lease, /*byte_length=*/32, &handle));

  iree_net_rdma_send_reservation_table_deinitialize(&table_);
  EXPECT_EQ(1u, release_count);
}

TEST_F(SendReservationTableTest, ReportsCapacityExhaustion) {
  IREE_ASSERT_OK(Initialize(1));

  uint32_t release_count = 0;
  iree_async_buffer_lease_t first_lease = MakeLease(&release_count);
  iree_net_carrier_send_handle_t first_handle = 0;
  IREE_ASSERT_OK(iree_net_rdma_send_reservation_table_acquire(
      &table_, &first_lease, /*byte_length=*/32, &first_handle));

  iree_async_buffer_lease_t second_lease = MakeLease(&release_count);
  iree_net_carrier_send_handle_t second_handle = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_net_rdma_send_reservation_table_acquire(
          &table_, &second_lease, /*byte_length=*/32, &second_handle));
  EXPECT_NE(nullptr, second_lease.release.fn);
  iree_async_buffer_lease_release(&second_lease);
}

TEST_F(SendReservationTableTest, RejectsStaleAndDoubleResolve) {
  IREE_ASSERT_OK(Initialize(1));

  uint32_t release_count = 0;
  iree_async_buffer_lease_t first_lease = MakeLease(&release_count);
  iree_net_carrier_send_handle_t first_handle = 0;
  IREE_ASSERT_OK(iree_net_rdma_send_reservation_table_acquire(
      &table_, &first_lease, /*byte_length=*/32, &first_handle));

  iree_net_rdma_send_reservation_t reservation;
  IREE_ASSERT_OK(iree_net_rdma_send_reservation_table_resolve(
      &table_, first_handle, &reservation));
  iree_async_buffer_lease_release(&reservation.buffer_lease);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_net_rdma_send_reservation_table_resolve(
                            &table_, first_handle, &reservation));

  iree_async_buffer_lease_t second_lease = MakeLease(&release_count);
  iree_net_carrier_send_handle_t second_handle = 0;
  IREE_ASSERT_OK(iree_net_rdma_send_reservation_table_acquire(
      &table_, &second_lease, /*byte_length=*/16, &second_handle));
  EXPECT_NE(first_handle, second_handle);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_net_rdma_send_reservation_table_resolve(
                            &table_, first_handle, &reservation));
}

TEST_F(SendReservationTableTest, RejectsInvalidArguments) {
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Initialize(0));

  uint32_t release_count = 0;
  iree_async_buffer_lease_t lease = MakeLease(&release_count);
  iree_net_carrier_send_handle_t handle = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_net_rdma_send_reservation_table_acquire(
                            &table_, &lease,
                            /*byte_length=*/32, &handle));

  IREE_ASSERT_OK(Initialize(1));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_net_rdma_send_reservation_table_acquire(
                            &table_, nullptr,
                            /*byte_length=*/32, &handle));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_rdma_send_reservation_table_acquire(&table_, &lease,
                                                   /*byte_length=*/0, &handle));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        iree_net_rdma_send_reservation_table_acquire(
                            &table_, &lease,
                            /*byte_length=*/129, &handle));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_net_rdma_send_reservation_table_acquire(
                            &table_, &lease,
                            /*byte_length=*/32, nullptr));

  iree_net_rdma_send_reservation_t reservation;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_net_rdma_send_reservation_table_resolve(
                            &table_, UINT64_MAX, &reservation));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_rdma_send_reservation_table_resolve(&table_, 0u, nullptr));
  iree_async_buffer_lease_release(&lease);
}

TEST(SendReservationTableStandaloneTest, NullTableHasNoAvailableCapacity) {
  EXPECT_EQ(0u,
            iree_net_rdma_send_reservation_table_available_capacity(nullptr));
}

}  // namespace
