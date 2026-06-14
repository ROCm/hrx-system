// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/carrier/rdma/receive_queue.h"

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

static void DestroyTestRegion(iree_async_region_t* region) {
  iree_allocator_free(iree_allocator_system(), region);
}

class ReceiveQueueTest : public ::testing::Test {
 protected:
  void SetUp() override {
    storage_.resize(kBufferSize * kBufferCount);

    iree_async_region_t* region = nullptr;
    IREE_ASSERT_OK(iree_allocator_malloc(iree_allocator_system(),
                                         sizeof(*region), (void**)&region));
    std::memset(region, 0, sizeof(*region));
    iree_atomic_ref_count_init(&region->ref_count);
    region->destroy_fn = DestroyTestRegion;
    region->type = IREE_ASYNC_REGION_TYPE_RDMA;
    region->base_ptr = storage_.data();
    region->length = storage_.size();
    region->buffer_size = kBufferSize;
    region->buffer_count = kBufferCount;
    region->handles.rdma.lkey = kLocalKey;
    test_region_ = region;

    IREE_ASSERT_OK(iree_async_buffer_pool_create(
        test_region_, iree_allocator_system(), &buffer_pool_));

    std::memset(&verbs_context_, 0, sizeof(verbs_context_));
    verbs_context_.ops.post_recv = PostRecvThunk;
    std::memset(&native_queue_pair_, 0, sizeof(native_queue_pair_));
    native_queue_pair_.context = &verbs_context_;
    native_queue_pair_.qp_context = this;
    std::memset(&queue_pair_, 0, sizeof(queue_pair_));
    queue_pair_.native_qp = &native_queue_pair_;

    IREE_ASSERT_OK(iree_net_rdma_work_request_table_initialize(
        kBufferCount, iree_allocator_system(), &work_request_table_));
  }

  void TearDown() override {
    iree_net_rdma_receive_queue_deinitialize(&receive_queue_);
    iree_net_rdma_work_request_table_deinitialize(&work_request_table_);
    iree_async_buffer_pool_release(buffer_pool_);
    iree_async_region_release(test_region_);
  }

  iree_status_t InitializeReceiveQueue(uint32_t capacity = kBufferCount) {
    return iree_net_rdma_receive_queue_initialize(
        &queue_pair_, &work_request_table_, buffer_pool_, capacity,
        iree_allocator_system(), &receive_queue_);
  }

  static int PostRecvThunk(struct ibv_qp* queue_pair, struct ibv_recv_wr* wr,
                           struct ibv_recv_wr** bad_wr) {
    auto* test = static_cast<ReceiveQueueTest*>(queue_pair->qp_context);
    return test->PostRecv(wr, bad_wr);
  }

  int PostRecv(struct ibv_recv_wr* wr, struct ibv_recv_wr** bad_wr) {
    if (post_results_.size() >= fail_after_) {
      if (bad_wr) *bad_wr = wr;
      return fail_error_;
    }

    EXPECT_EQ(nullptr, wr->next);
    EXPECT_EQ(1, wr->num_sge);
    EXPECT_NE(nullptr, wr->sg_list);
    if (wr->sg_list) {
      EXPECT_EQ(kBufferSize, wr->sg_list[0].length);
      EXPECT_EQ(kLocalKey, wr->sg_list[0].lkey);
    }

    struct ibv_sge scatter_gather_entry = {};
    if (wr->sg_list) {
      scatter_gather_entry = wr->sg_list[0];
    }
    post_results_.push_back({wr->wr_id, scatter_gather_entry});
    if (bad_wr) *bad_wr = nullptr;
    return 0;
  }

  struct PostResult {
    uint64_t work_request_id;
    struct ibv_sge scatter_gather_entry;
  };

  static constexpr uint32_t kBufferSize = 64;
  static constexpr uint32_t kBufferCount = 4;
  static constexpr uint32_t kLocalKey = 0xABCD1234u;

  std::vector<char> storage_;
  iree_async_region_t* test_region_ = nullptr;
  iree_async_buffer_pool_t* buffer_pool_ = nullptr;
  struct ibv_context verbs_context_ = {};
  struct ibv_qp native_queue_pair_ = {};
  iree_net_rdma_queue_pair_t queue_pair_ = {};
  iree_net_rdma_work_request_table_t work_request_table_ = {};
  iree_net_rdma_receive_queue_t receive_queue_ = {};
  std::vector<PostResult> post_results_;
  size_t fail_after_ = SIZE_MAX;
  int fail_error_ = ENOMEM;
};

TEST_F(ReceiveQueueTest, ReplenishPostsReceives) {
  IREE_ASSERT_OK(InitializeReceiveQueue());

  uint32_t posted_count = 0;
  IREE_ASSERT_OK(iree_net_rdma_receive_queue_replenish(
      &receive_queue_, /*target_posted_count=*/3, &posted_count));

  EXPECT_EQ(3u, posted_count);
  EXPECT_EQ(3u, iree_net_rdma_receive_queue_posted_count(&receive_queue_));
  EXPECT_EQ(1u,
            iree_net_rdma_receive_queue_available_capacity(&receive_queue_));
  EXPECT_EQ(1u, iree_async_buffer_pool_available(buffer_pool_));
  EXPECT_EQ(1u, iree_net_rdma_work_request_table_available_capacity(
                    &work_request_table_));
  ASSERT_EQ(3u, post_results_.size());
}

TEST_F(ReceiveQueueTest, CompletionReturnsLease) {
  IREE_ASSERT_OK(InitializeReceiveQueue());

  uint32_t posted_count = 0;
  IREE_ASSERT_OK(iree_net_rdma_receive_queue_replenish(
      &receive_queue_, /*target_posted_count=*/1, &posted_count));
  ASSERT_EQ(1u, post_results_.size());

  iree_net_rdma_work_request_completion_t completion;
  IREE_ASSERT_OK(iree_net_rdma_work_request_table_complete(
      &work_request_table_, post_results_[0].work_request_id, &completion));

  iree_async_buffer_lease_t lease;
  IREE_ASSERT_OK(iree_net_rdma_receive_queue_complete(
      &receive_queue_, completion, /*byte_length=*/7, &lease));

  EXPECT_EQ(0u, iree_net_rdma_receive_queue_posted_count(&receive_queue_));
  EXPECT_EQ(kBufferCount,
            iree_net_rdma_receive_queue_available_capacity(&receive_queue_));
  EXPECT_EQ(kBufferCount - 1, iree_async_buffer_pool_available(buffer_pool_));
  EXPECT_EQ(7u, lease.span.length);

  iree_async_buffer_lease_release(&lease);
  EXPECT_EQ(kBufferCount, iree_async_buffer_pool_available(buffer_pool_));
}

TEST_F(ReceiveQueueTest, ReplenishFailureDoesNotLeakAttemptedPost) {
  IREE_ASSERT_OK(InitializeReceiveQueue());
  fail_after_ = 0;
  fail_error_ = ENOMEM;

  uint32_t posted_count = 1;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_net_rdma_receive_queue_replenish(
          &receive_queue_, /*target_posted_count=*/1, &posted_count));

  EXPECT_EQ(0u, posted_count);
  EXPECT_EQ(0u, iree_net_rdma_receive_queue_posted_count(&receive_queue_));
  EXPECT_EQ(kBufferCount,
            iree_net_rdma_receive_queue_available_capacity(&receive_queue_));
  EXPECT_EQ(kBufferCount, iree_async_buffer_pool_available(buffer_pool_));
  EXPECT_EQ(kBufferCount, iree_net_rdma_work_request_table_available_capacity(
                              &work_request_table_));
}

TEST_F(ReceiveQueueTest, ReplenishFailureLeavesEarlierPostsOwned) {
  IREE_ASSERT_OK(InitializeReceiveQueue());
  fail_after_ = 2;
  fail_error_ = ENOMEM;

  uint32_t posted_count = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_net_rdma_receive_queue_replenish(
          &receive_queue_, /*target_posted_count=*/3, &posted_count));

  EXPECT_EQ(2u, posted_count);
  EXPECT_EQ(2u, iree_net_rdma_receive_queue_posted_count(&receive_queue_));
  EXPECT_EQ(kBufferCount - 2,
            iree_net_rdma_receive_queue_available_capacity(&receive_queue_));
  EXPECT_EQ(kBufferCount - 2, iree_async_buffer_pool_available(buffer_pool_));
  EXPECT_EQ(kBufferCount - 2,
            iree_net_rdma_work_request_table_available_capacity(
                &work_request_table_));
}

TEST_F(ReceiveQueueTest, ReplenishStopsWhenBufferPoolIsExhausted) {
  IREE_ASSERT_OK(InitializeReceiveQueue());

  iree_async_buffer_lease_t external_leases[kBufferCount];
  std::memset(external_leases, 0, sizeof(external_leases));
  for (uint32_t i = 0; i < kBufferCount; ++i) {
    IREE_ASSERT_OK(
        iree_async_buffer_pool_acquire(buffer_pool_, &external_leases[i]));
  }

  uint32_t posted_count = 1;
  IREE_ASSERT_OK(iree_net_rdma_receive_queue_replenish(
      &receive_queue_, /*target_posted_count=*/1, &posted_count));

  EXPECT_EQ(0u, posted_count);
  EXPECT_EQ(0u, iree_net_rdma_receive_queue_posted_count(&receive_queue_));
  EXPECT_EQ(kBufferCount,
            iree_net_rdma_receive_queue_available_capacity(&receive_queue_));
  EXPECT_EQ(0u, iree_async_buffer_pool_available(buffer_pool_));
  EXPECT_EQ(kBufferCount, iree_net_rdma_work_request_table_available_capacity(
                              &work_request_table_));

  for (uint32_t i = 0; i < kBufferCount; ++i) {
    iree_async_buffer_lease_release(&external_leases[i]);
  }
}

TEST_F(ReceiveQueueTest, CompleteRejectsInvalidByteLength) {
  IREE_ASSERT_OK(InitializeReceiveQueue());

  uint32_t posted_count = 0;
  IREE_ASSERT_OK(iree_net_rdma_receive_queue_replenish(
      &receive_queue_, /*target_posted_count=*/1, &posted_count));
  ASSERT_EQ(1u, post_results_.size());

  iree_net_rdma_work_request_completion_t completion;
  IREE_ASSERT_OK(iree_net_rdma_work_request_table_complete(
      &work_request_table_, post_results_[0].work_request_id, &completion));

  iree_async_buffer_lease_t lease;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        iree_net_rdma_receive_queue_complete(
                            &receive_queue_, completion,
                            /*byte_length=*/kBufferSize + 1, &lease));

  EXPECT_EQ(0u, iree_net_rdma_receive_queue_posted_count(&receive_queue_));
  EXPECT_EQ(kBufferCount, iree_async_buffer_pool_available(buffer_pool_));
}

TEST_F(ReceiveQueueTest, RejectsInvalidArguments) {
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_rdma_receive_queue_initialize(
          nullptr, &work_request_table_, buffer_pool_, kBufferCount,
          iree_allocator_system(), &receive_queue_));

  IREE_ASSERT_OK(InitializeReceiveQueue());
  uint32_t posted_count = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_net_rdma_receive_queue_replenish(
                            &receive_queue_, kBufferCount + 1, &posted_count));

  iree_async_buffer_lease_t lease;
  iree_net_rdma_work_request_completion_t completion = {
      IREE_NET_RDMA_WORK_REQUEST_OPERATION_SEND, 0};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_rdma_receive_queue_complete(&receive_queue_, completion,
                                           /*byte_length=*/0, &lease));
}

TEST(ReceiveQueueStandaloneTest, NullQueriesReturnZero) {
  EXPECT_EQ(0u, iree_net_rdma_receive_queue_posted_count(nullptr));
  EXPECT_EQ(0u, iree_net_rdma_receive_queue_available_capacity(nullptr));
}

}  // namespace
