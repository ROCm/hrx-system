// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/carrier/rdma/work_request_table.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

static void CountLeaseRelease(void* user_data, uint32_t buffer_index) {
  uint32_t* release_count = (uint32_t*)user_data;
  EXPECT_EQ(7u, buffer_index);
  *release_count += 1;
}

class WorkRequestTableTest : public ::testing::Test {
 protected:
  void TearDown() override {
    iree_net_rdma_work_request_table_deinitialize(&table_);
  }

  iree_status_t Initialize(uint32_t capacity) {
    return iree_net_rdma_work_request_table_initialize(
        capacity, iree_allocator_system(), &table_);
  }

  iree_net_rdma_work_request_table_t table_ = {};
};

TEST_F(WorkRequestTableTest, AcquireAndCompleteReturnsMetadata) {
  IREE_ASSERT_OK(Initialize(2));

  uint64_t wr_id = 0;
  IREE_ASSERT_OK(iree_net_rdma_work_request_table_acquire(
      &table_, IREE_NET_RDMA_WORK_REQUEST_OPERATION_SEND, 0xABCDu,
      /*byte_length=*/128, /*retained_buffer_lease=*/nullptr, &wr_id));
  EXPECT_EQ(1u, iree_net_rdma_work_request_table_available_capacity(&table_));

  iree_net_rdma_work_request_completion_t completion;
  IREE_ASSERT_OK(
      iree_net_rdma_work_request_table_complete(&table_, wr_id, &completion));
  EXPECT_EQ(IREE_NET_RDMA_WORK_REQUEST_OPERATION_SEND, completion.operation);
  EXPECT_EQ(0xABCDu, completion.user_data);
  EXPECT_EQ(128u, completion.byte_length);
  EXPECT_EQ(2u, iree_net_rdma_work_request_table_available_capacity(&table_));
}

TEST_F(WorkRequestTableTest, ReportsCapacityExhaustion) {
  IREE_ASSERT_OK(Initialize(1));

  uint64_t wr_id = 0;
  IREE_ASSERT_OK(iree_net_rdma_work_request_table_acquire(
      &table_, IREE_NET_RDMA_WORK_REQUEST_OPERATION_RECV, 1u,
      /*byte_length=*/0, /*retained_buffer_lease=*/nullptr, &wr_id));
  EXPECT_EQ(0u, iree_net_rdma_work_request_table_available_capacity(&table_));

  uint64_t unused_wr_id = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        iree_net_rdma_work_request_table_acquire(
                            &table_, IREE_NET_RDMA_WORK_REQUEST_OPERATION_RECV,
                            2u, /*byte_length=*/0,
                            /*retained_buffer_lease=*/nullptr, &unused_wr_id));
}

TEST_F(WorkRequestTableTest, RejectsStaleAndDoubleCompletions) {
  IREE_ASSERT_OK(Initialize(1));

  uint64_t first_wr_id = 0;
  IREE_ASSERT_OK(iree_net_rdma_work_request_table_acquire(
      &table_, IREE_NET_RDMA_WORK_REQUEST_OPERATION_DIRECT_WRITE, 1u,
      /*byte_length=*/4, /*retained_buffer_lease=*/nullptr, &first_wr_id));

  iree_net_rdma_work_request_completion_t completion;
  IREE_ASSERT_OK(iree_net_rdma_work_request_table_complete(&table_, first_wr_id,
                                                           &completion));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_net_rdma_work_request_table_complete(
                            &table_, first_wr_id, &completion));

  uint64_t second_wr_id = 0;
  IREE_ASSERT_OK(iree_net_rdma_work_request_table_acquire(
      &table_, IREE_NET_RDMA_WORK_REQUEST_OPERATION_DIRECT_READ, 2u,
      /*byte_length=*/8, /*retained_buffer_lease=*/nullptr, &second_wr_id));
  EXPECT_NE(first_wr_id, second_wr_id);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_net_rdma_work_request_table_complete(
                            &table_, first_wr_id, &completion));
  IREE_ASSERT_OK(iree_net_rdma_work_request_table_complete(
      &table_, second_wr_id, &completion));
  EXPECT_EQ(IREE_NET_RDMA_WORK_REQUEST_OPERATION_DIRECT_READ,
            completion.operation);
  EXPECT_EQ(2u, completion.user_data);
  EXPECT_EQ(8u, completion.byte_length);
}

TEST_F(WorkRequestTableTest, RejectsInvalidArguments) {
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Initialize(0));

  uint64_t wr_id = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_rdma_work_request_table_acquire(
          &table_, IREE_NET_RDMA_WORK_REQUEST_OPERATION_NONE, 0u,
          /*byte_length=*/0, /*retained_buffer_lease=*/nullptr, &wr_id));

  IREE_ASSERT_OK(Initialize(1));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_rdma_work_request_table_acquire(
          &table_, IREE_NET_RDMA_WORK_REQUEST_OPERATION_NONE, 0u,
          /*byte_length=*/0, /*retained_buffer_lease=*/nullptr, &wr_id));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_rdma_work_request_table_acquire(
          &table_, (iree_net_rdma_work_request_operation_t)255u, 0u,
          /*byte_length=*/0, /*retained_buffer_lease=*/nullptr, &wr_id));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_rdma_work_request_table_acquire(
          &table_, IREE_NET_RDMA_WORK_REQUEST_OPERATION_SEND, 0u,
          /*byte_length=*/0, /*retained_buffer_lease=*/nullptr, nullptr));

  iree_net_rdma_work_request_completion_t completion;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_net_rdma_work_request_table_complete(
                            &table_, UINT64_MAX, &completion));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_rdma_work_request_table_complete(&table_, 0u, nullptr));
}

TEST_F(WorkRequestTableTest, RetainsLeaseUntilCompletion) {
  IREE_ASSERT_OK(Initialize(1));

  uint32_t release_count = 0;
  iree_async_buffer_lease_t lease = {};
  lease.release = (iree_async_buffer_recycle_callback_t){
      .fn = CountLeaseRelease,
      .user_data = &release_count,
  };
  lease.buffer_index = 7u;

  uint64_t wr_id = 0;
  IREE_ASSERT_OK(iree_net_rdma_work_request_table_acquire(
      &table_, IREE_NET_RDMA_WORK_REQUEST_OPERATION_SEND, 0x1234u,
      /*byte_length=*/256, &lease, &wr_id));
  EXPECT_EQ(nullptr, lease.release.fn);
  EXPECT_EQ(0u, release_count);

  iree_net_rdma_work_request_completion_t completion;
  IREE_ASSERT_OK(
      iree_net_rdma_work_request_table_complete(&table_, wr_id, &completion));
  EXPECT_EQ(IREE_NET_RDMA_WORK_REQUEST_OPERATION_SEND, completion.operation);
  EXPECT_EQ(0x1234u, completion.user_data);
  EXPECT_EQ(256u, completion.byte_length);
  EXPECT_NE(nullptr, completion.retained_buffer_lease.release.fn);

  iree_async_buffer_lease_release(&completion.retained_buffer_lease);
  iree_async_buffer_lease_release(&completion.retained_buffer_lease);
  EXPECT_EQ(1u, release_count);
}

TEST(WorkRequestTableStandaloneTest, NullTableHasNoAvailableCapacity) {
  EXPECT_EQ(0u, iree_net_rdma_work_request_table_available_capacity(nullptr));
}

}  // namespace
