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
      /*.fn=*/CountLeaseRelease,
      /*.user_data=*/release_count,
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

  iree_status_t Acquire(iree_async_buffer_lease_t* lease,
                        iree_host_size_t byte_length,
                        iree_net_carrier_send_handle_t* out_handle) {
    iree_async_span_t span = iree_async_span_make(
        lease->span.region, lease->span.offset, byte_length);
    return iree_net_rdma_send_reservation_table_acquire(
        &table_, iree_async_span_list_make(&span, 1), lease,
        IREE_NET_RDMA_SEND_RESERVATION_COMPLETION_INTERNAL,
        /*user_data=*/0, out_handle);
  }

  iree_async_buffer_lease_t MakeLease() { return ::MakeLease(&release_count_); }

  iree_net_rdma_send_reservation_table_t table_ = {};
  uint32_t release_count_ = 0;
};

TEST_F(SendReservationTableTest, AcquireAndResolveTransfersLease) {
  IREE_ASSERT_OK(Initialize(2));

  iree_async_buffer_lease_t lease = MakeLease();
  iree_net_carrier_send_handle_t handle = 0;
  IREE_ASSERT_OK(Acquire(&lease, /*byte_length=*/64, &handle));
  EXPECT_EQ(nullptr, lease.release.fn);
  EXPECT_NE(0u, handle);
  EXPECT_EQ(1u,
            iree_net_rdma_send_reservation_table_available_capacity(&table_));

  iree_net_rdma_send_reservation_t reservation;
  IREE_ASSERT_OK(iree_net_rdma_send_reservation_table_resolve(&table_, handle,
                                                              &reservation));
  EXPECT_EQ(64u, reservation.byte_length);
  EXPECT_EQ(1u, reservation.span_count);
  EXPECT_EQ(64u, reservation.spans[0].length);
  EXPECT_NE(nullptr, reservation.buffer_lease.release.fn);
  EXPECT_EQ(2u,
            iree_net_rdma_send_reservation_table_available_capacity(&table_));

  iree_async_buffer_lease_release(&reservation.buffer_lease);
  iree_async_buffer_lease_release(&reservation.buffer_lease);
  EXPECT_EQ(1u, release_count_);
}

TEST_F(SendReservationTableTest, AbortReleasesLease) {
  IREE_ASSERT_OK(Initialize(1));

  iree_async_buffer_lease_t lease = MakeLease();
  iree_net_carrier_send_handle_t handle = 0;
  IREE_ASSERT_OK(Acquire(&lease, /*byte_length=*/32, &handle));

  IREE_ASSERT_OK(iree_net_rdma_send_reservation_table_abort(&table_, handle));
  EXPECT_EQ(1u, release_count_);
  EXPECT_EQ(1u,
            iree_net_rdma_send_reservation_table_available_capacity(&table_));
}

TEST_F(SendReservationTableTest, CommitQueuesPendingReservationsInOrder) {
  IREE_ASSERT_OK(Initialize(2));

  iree_async_buffer_lease_t first_lease = MakeLease();
  iree_net_carrier_send_handle_t first_handle = 0;
  IREE_ASSERT_OK(Acquire(&first_lease, /*byte_length=*/32, &first_handle));
  EXPECT_NE(0u, first_handle);
  iree_async_buffer_lease_t second_lease = MakeLease();
  iree_net_carrier_send_handle_t second_handle = 0;
  IREE_ASSERT_OK(Acquire(&second_lease, /*byte_length=*/48, &second_handle));

  IREE_ASSERT_OK(
      iree_net_rdma_send_reservation_table_commit(&table_, first_handle));
  IREE_ASSERT_OK(
      iree_net_rdma_send_reservation_table_commit(&table_, second_handle));
  EXPECT_EQ(0u,
            iree_net_rdma_send_reservation_table_available_capacity(&table_));
  EXPECT_EQ(2u, iree_net_rdma_send_reservation_table_pending_count(&table_));

  iree_net_carrier_send_handle_t pending_handle = 0;
  iree_net_rdma_send_reservation_t reservation;
  IREE_ASSERT_OK(iree_net_rdma_send_reservation_table_peek_pending_front(
      &table_, &pending_handle, &reservation));
  EXPECT_EQ(first_handle, pending_handle);
  EXPECT_EQ(32u, reservation.byte_length);
  EXPECT_EQ(0u, release_count_);

  IREE_ASSERT_OK(iree_net_rdma_send_reservation_table_resolve_pending_front(
      &table_, &pending_handle, &reservation));
  EXPECT_EQ(first_handle, pending_handle);
  EXPECT_EQ(32u, reservation.byte_length);
  EXPECT_NE(nullptr, reservation.buffer_lease.release.fn);
  EXPECT_EQ(1u,
            iree_net_rdma_send_reservation_table_available_capacity(&table_));
  EXPECT_EQ(1u, iree_net_rdma_send_reservation_table_pending_count(&table_));
  iree_async_buffer_lease_release(&reservation.buffer_lease);

  IREE_ASSERT_OK(iree_net_rdma_send_reservation_table_resolve_pending_front(
      &table_, &pending_handle, &reservation));
  EXPECT_EQ(second_handle, pending_handle);
  EXPECT_EQ(48u, reservation.byte_length);
  EXPECT_EQ(2u,
            iree_net_rdma_send_reservation_table_available_capacity(&table_));
  EXPECT_EQ(0u, iree_net_rdma_send_reservation_table_pending_count(&table_));
  iree_async_buffer_lease_release(&reservation.buffer_lease);
  EXPECT_EQ(2u, release_count_);
}

TEST_F(SendReservationTableTest, CommitConsumesCallerHandle) {
  IREE_ASSERT_OK(Initialize(1));

  iree_async_buffer_lease_t lease = MakeLease();
  iree_net_carrier_send_handle_t handle = 0;
  IREE_ASSERT_OK(Acquire(&lease, /*byte_length=*/32, &handle));
  IREE_ASSERT_OK(iree_net_rdma_send_reservation_table_commit(&table_, handle));

  iree_net_rdma_send_reservation_t reservation;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_net_rdma_send_reservation_table_commit(&table_, handle));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_net_rdma_send_reservation_table_peek(&table_, handle, &reservation));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_net_rdma_send_reservation_table_resolve(
                            &table_, handle, &reservation));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_net_rdma_send_reservation_table_abort(&table_, handle));
  EXPECT_EQ(0u, release_count_);

  IREE_ASSERT_OK(iree_net_rdma_send_reservation_table_resolve_pending_front(
      &table_, nullptr, &reservation));
  iree_async_buffer_lease_release(&reservation.buffer_lease);
}

TEST_F(SendReservationTableTest, PreservesCompletionMetadata) {
  IREE_ASSERT_OK(Initialize(1));

  iree_async_buffer_lease_t lease = MakeLease();
  iree_async_span_t span =
      iree_async_span_make(lease.span.region, lease.span.offset, 32);
  iree_net_carrier_send_handle_t handle = 0;
  IREE_ASSERT_OK(iree_net_rdma_send_reservation_table_acquire(
      &table_, iree_async_span_list_make(&span, 1), &lease,
      IREE_NET_RDMA_SEND_RESERVATION_COMPLETION_SEND, /*user_data=*/0x1234u,
      &handle));
  IREE_ASSERT_OK(iree_net_rdma_send_reservation_table_commit(&table_, handle));

  iree_net_rdma_send_reservation_t reservation;
  IREE_ASSERT_OK(iree_net_rdma_send_reservation_table_resolve_pending_front(
      &table_, nullptr, &reservation));
  EXPECT_EQ(32u, reservation.byte_length);
  EXPECT_EQ(IREE_NET_RDMA_SEND_RESERVATION_COMPLETION_SEND,
            reservation.completion);
  EXPECT_EQ(0x1234u, reservation.user_data);

  iree_async_buffer_lease_release(&reservation.buffer_lease);
  EXPECT_EQ(1u, release_count_);
}

TEST_F(SendReservationTableTest, MixedReservationsRetainLeaseAndSpanMetadata) {
  IREE_ASSERT_OK(Initialize(1));

  iree_async_buffer_lease_t lease = MakeLease();
  iree_async_span_t spans[2] = {
      iree_async_span_from_ptr(reinterpret_cast<void*>(0x1000), 32),
      iree_async_span_from_ptr(reinterpret_cast<void*>(0x2000), 64),
  };
  iree_net_carrier_send_handle_t handle = 0;
  IREE_ASSERT_OK(iree_net_rdma_send_reservation_table_acquire(
      &table_, iree_async_span_list_make(spans, IREE_ARRAYSIZE(spans)), &lease,
      IREE_NET_RDMA_SEND_RESERVATION_COMPLETION_SEND,
      /*user_data=*/0x5678u, &handle));
  EXPECT_EQ(nullptr, lease.release.fn);
  spans[0] = iree_async_span_empty();
  spans[1] = iree_async_span_empty();

  IREE_ASSERT_OK(iree_net_rdma_send_reservation_table_commit(&table_, handle));
  iree_net_rdma_send_reservation_t reservation;
  IREE_ASSERT_OK(iree_net_rdma_send_reservation_table_resolve_pending_front(
      &table_, nullptr, &reservation));
  EXPECT_EQ(2u, reservation.span_count);
  EXPECT_EQ(32u, reservation.spans[0].length);
  EXPECT_EQ(64u, reservation.spans[1].length);
  EXPECT_EQ(96u, reservation.byte_length);
  EXPECT_EQ(IREE_NET_RDMA_SEND_RESERVATION_COMPLETION_SEND,
            reservation.completion);
  EXPECT_EQ(0x5678u, reservation.user_data);
  EXPECT_NE(nullptr, reservation.buffer_lease.release.fn);
  iree_async_buffer_lease_release(&reservation.buffer_lease);
  EXPECT_EQ(1u, release_count_);
}

TEST_F(SendReservationTableTest, RegisteredReservationNeedsNoLease) {
  IREE_ASSERT_OK(Initialize(1));

  iree_async_span_t spans[2] = {
      iree_async_span_from_ptr(reinterpret_cast<void*>(0x1000), 32),
      iree_async_span_from_ptr(reinterpret_cast<void*>(0x2000), 64),
  };
  iree_net_carrier_send_handle_t handle = 0;
  IREE_ASSERT_OK(iree_net_rdma_send_reservation_table_acquire(
      &table_, iree_async_span_list_make(spans, IREE_ARRAYSIZE(spans)),
      /*buffer_lease=*/nullptr, IREE_NET_RDMA_SEND_RESERVATION_COMPLETION_SEND,
      /*user_data=*/0x5678u, &handle));
  spans[0] = iree_async_span_empty();
  spans[1] = iree_async_span_empty();

  IREE_ASSERT_OK(iree_net_rdma_send_reservation_table_commit(&table_, handle));
  iree_net_rdma_send_reservation_t reservation;
  IREE_ASSERT_OK(iree_net_rdma_send_reservation_table_resolve_pending_front(
      &table_, nullptr, &reservation));
  EXPECT_EQ(2u, reservation.span_count);
  EXPECT_EQ(32u, reservation.spans[0].length);
  EXPECT_EQ(64u, reservation.spans[1].length);
  EXPECT_EQ(nullptr, reservation.buffer_lease.release.fn);
}

TEST_F(SendReservationTableTest, AbortingMixedReservationReleasesLease) {
  IREE_ASSERT_OK(Initialize(1));

  iree_async_buffer_lease_t lease = MakeLease();
  iree_async_span_t spans[2] = {
      iree_async_span_from_ptr(reinterpret_cast<void*>(0x1000), 32),
      iree_async_span_from_ptr(reinterpret_cast<void*>(0x2000), 64),
  };
  iree_net_carrier_send_handle_t handle = 0;
  IREE_ASSERT_OK(iree_net_rdma_send_reservation_table_acquire(
      &table_, iree_async_span_list_make(spans, IREE_ARRAYSIZE(spans)), &lease,
      IREE_NET_RDMA_SEND_RESERVATION_COMPLETION_SEND,
      /*user_data=*/0x5678u, &handle));
  IREE_ASSERT_OK(iree_net_rdma_send_reservation_table_commit(&table_, handle));

  uint32_t aborted_count = 0;
  IREE_ASSERT_OK(
      iree_net_rdma_send_reservation_table_abort_all(&table_, &aborted_count));
  EXPECT_EQ(1u, aborted_count);
  EXPECT_EQ(1u, release_count_);
}

TEST_F(SendReservationTableTest, AbortAllReleasesAllLeases) {
  IREE_ASSERT_OK(Initialize(3));

  iree_async_buffer_lease_t first_lease = MakeLease();
  iree_net_carrier_send_handle_t first_handle = 0;
  IREE_ASSERT_OK(Acquire(&first_lease, /*byte_length=*/32, &first_handle));
  EXPECT_NE(0u, first_handle);
  iree_async_buffer_lease_t second_lease = MakeLease();
  iree_net_carrier_send_handle_t second_handle = 0;
  IREE_ASSERT_OK(Acquire(&second_lease, /*byte_length=*/48, &second_handle));
  IREE_ASSERT_OK(
      iree_net_rdma_send_reservation_table_commit(&table_, second_handle));

  uint32_t aborted_count = 0;
  IREE_ASSERT_OK(
      iree_net_rdma_send_reservation_table_abort_all(&table_, &aborted_count));
  EXPECT_EQ(2u, aborted_count);
  EXPECT_EQ(2u, release_count_);
  EXPECT_EQ(3u,
            iree_net_rdma_send_reservation_table_available_capacity(&table_));
  EXPECT_EQ(0u, iree_net_rdma_send_reservation_table_pending_count(&table_));

  iree_net_rdma_send_reservation_t reservation;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_net_rdma_send_reservation_table_resolve(
                            &table_, first_handle, &reservation));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_net_rdma_send_reservation_table_resolve(
                            &table_, second_handle, &reservation));
}

TEST_F(SendReservationTableTest, ResolveNextDrainsActiveReservations) {
  IREE_ASSERT_OK(Initialize(3));

  iree_async_buffer_lease_t first_lease = MakeLease();
  iree_net_carrier_send_handle_t first_handle = 0;
  IREE_ASSERT_OK(Acquire(&first_lease, /*byte_length=*/32, &first_handle));
  iree_async_buffer_lease_t second_lease = MakeLease();
  iree_async_span_t second_span = iree_async_span_make(
      second_lease.span.region, second_lease.span.offset, 48);
  iree_net_carrier_send_handle_t second_handle = 0;
  IREE_ASSERT_OK(iree_net_rdma_send_reservation_table_acquire(
      &table_, iree_async_span_list_make(&second_span, 1), &second_lease,
      IREE_NET_RDMA_SEND_RESERVATION_COMPLETION_SEND, /*user_data=*/0xABCDu,
      &second_handle));
  IREE_ASSERT_OK(
      iree_net_rdma_send_reservation_table_commit(&table_, second_handle));

  uint32_t cursor = 0;
  bool found = false;
  iree_net_rdma_send_reservation_t reservation;
  IREE_ASSERT_OK(iree_net_rdma_send_reservation_table_resolve_next(
      &table_, &cursor, &reservation, &found));
  EXPECT_TRUE(found);
  EXPECT_EQ(32u, reservation.byte_length);
  EXPECT_EQ(IREE_NET_RDMA_SEND_RESERVATION_COMPLETION_INTERNAL,
            reservation.completion);
  EXPECT_EQ(0u, reservation.user_data);
  EXPECT_EQ(1u, iree_net_rdma_send_reservation_table_pending_count(&table_));
  iree_async_buffer_lease_release(&reservation.buffer_lease);

  IREE_ASSERT_OK(iree_net_rdma_send_reservation_table_resolve_next(
      &table_, &cursor, &reservation, &found));
  EXPECT_TRUE(found);
  EXPECT_EQ(48u, reservation.byte_length);
  EXPECT_EQ(IREE_NET_RDMA_SEND_RESERVATION_COMPLETION_SEND,
            reservation.completion);
  EXPECT_EQ(0xABCDu, reservation.user_data);
  EXPECT_EQ(0u, iree_net_rdma_send_reservation_table_pending_count(&table_));
  iree_async_buffer_lease_release(&reservation.buffer_lease);

  IREE_ASSERT_OK(iree_net_rdma_send_reservation_table_resolve_next(
      &table_, &cursor, &reservation, &found));
  EXPECT_FALSE(found);
  EXPECT_EQ(3u,
            iree_net_rdma_send_reservation_table_available_capacity(&table_));
  EXPECT_EQ(2u, release_count_);
}

TEST_F(SendReservationTableTest, DeinitializeReleasesOutstandingLeases) {
  IREE_ASSERT_OK(Initialize(1));

  iree_async_buffer_lease_t lease = MakeLease();
  iree_net_carrier_send_handle_t handle = 0;
  IREE_ASSERT_OK(Acquire(&lease, /*byte_length=*/32, &handle));

  iree_net_rdma_send_reservation_table_deinitialize(&table_);
  EXPECT_EQ(1u, release_count_);
}

TEST_F(SendReservationTableTest, ReportsCapacityExhaustion) {
  IREE_ASSERT_OK(Initialize(1));

  iree_async_buffer_lease_t first_lease = MakeLease();
  iree_net_carrier_send_handle_t first_handle = 0;
  IREE_ASSERT_OK(Acquire(&first_lease, /*byte_length=*/32, &first_handle));

  iree_async_buffer_lease_t second_lease = MakeLease();
  iree_net_carrier_send_handle_t second_handle = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      Acquire(&second_lease, /*byte_length=*/32, &second_handle));
  EXPECT_NE(nullptr, second_lease.release.fn);
  iree_async_buffer_lease_release(&second_lease);
}

TEST_F(SendReservationTableTest, RejectsStaleAndDoubleResolve) {
  IREE_ASSERT_OK(Initialize(1));

  iree_async_buffer_lease_t first_lease = MakeLease();
  iree_net_carrier_send_handle_t first_handle = 0;
  IREE_ASSERT_OK(Acquire(&first_lease, /*byte_length=*/32, &first_handle));

  iree_net_rdma_send_reservation_t reservation;
  IREE_ASSERT_OK(iree_net_rdma_send_reservation_table_resolve(
      &table_, first_handle, &reservation));
  iree_async_buffer_lease_release(&reservation.buffer_lease);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_net_rdma_send_reservation_table_resolve(
                            &table_, first_handle, &reservation));

  iree_async_buffer_lease_t second_lease = MakeLease();
  iree_net_carrier_send_handle_t second_handle = 0;
  IREE_ASSERT_OK(Acquire(&second_lease, /*byte_length=*/16, &second_handle));
  EXPECT_NE(first_handle, second_handle);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_net_rdma_send_reservation_table_resolve(
                            &table_, first_handle, &reservation));
}

TEST_F(SendReservationTableTest, RejectsInvalidArguments) {
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Initialize(0));

  iree_async_buffer_lease_t lease = MakeLease();
  iree_async_span_t spans[1] = {
      iree_async_span_from_ptr(reinterpret_cast<void*>(0x1000), 32),
  };
  iree_async_span_list_t span_list =
      iree_async_span_list_make(spans, IREE_ARRAYSIZE(spans));
  iree_net_carrier_send_handle_t handle = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_net_rdma_send_reservation_table_acquire(
                            &table_, span_list, &lease,
                            IREE_NET_RDMA_SEND_RESERVATION_COMPLETION_INTERNAL,
                            /*user_data=*/0, &handle));

  IREE_ASSERT_OK(Initialize(1));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_net_rdma_send_reservation_table_acquire(
                            &table_, iree_async_span_list_empty(), &lease,
                            IREE_NET_RDMA_SEND_RESERVATION_COMPLETION_INTERNAL,
                            /*user_data=*/0, &handle));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_rdma_send_reservation_table_acquire(
          &table_, iree_async_span_list_make(nullptr, 1), &lease,
          IREE_NET_RDMA_SEND_RESERVATION_COMPLETION_INTERNAL,
          /*user_data=*/0, &handle));
  iree_async_span_t empty_span = iree_async_span_empty();
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_rdma_send_reservation_table_acquire(
          &table_, iree_async_span_list_make(&empty_span, 1), &lease,
          IREE_NET_RDMA_SEND_RESERVATION_COMPLETION_INTERNAL,
          /*user_data=*/0, &handle));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_net_rdma_send_reservation_table_acquire(
                            &table_, span_list, &lease,
                            IREE_NET_RDMA_SEND_RESERVATION_COMPLETION_INTERNAL,
                            /*user_data=*/0, nullptr));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_net_rdma_send_reservation_table_acquire(
                            &table_, span_list, &lease,
                            (iree_net_rdma_send_reservation_completion_t)0xFFu,
                            /*user_data=*/0, &handle));

  iree_net_rdma_send_reservation_t reservation;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_net_rdma_send_reservation_table_resolve(
                            &table_, UINT64_MAX, &reservation));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_rdma_send_reservation_table_resolve(&table_, 0u, nullptr));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_rdma_send_reservation_table_peek(&table_, 0u, nullptr));
  uint32_t cursor = 0;
  bool found = false;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_net_rdma_send_reservation_table_resolve_next(
                            &table_, nullptr, &reservation, &found));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_net_rdma_send_reservation_table_resolve_next(
                            &table_, &cursor, nullptr, &found));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_net_rdma_send_reservation_table_resolve_next(
                            &table_, &cursor, &reservation, nullptr));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_net_rdma_send_reservation_table_peek_pending_front(
                            &table_, nullptr, &reservation));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_net_rdma_send_reservation_table_resolve_pending_front(
          &table_, nullptr, &reservation));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_net_rdma_send_reservation_table_peek_pending_front(
                            nullptr, nullptr, &reservation));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_rdma_send_reservation_table_resolve_pending_front(
          nullptr, nullptr, &reservation));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_net_rdma_send_reservation_table_abort_all(
                            nullptr, /*out_aborted_count=*/nullptr));
  iree_async_buffer_lease_release(&lease);
}

TEST(SendReservationTableStandaloneTest, NullTableHasNoAvailableCapacity) {
  EXPECT_EQ(0u,
            iree_net_rdma_send_reservation_table_available_capacity(nullptr));
  EXPECT_EQ(0u, iree_net_rdma_send_reservation_table_pending_count(nullptr));
}

}  // namespace
