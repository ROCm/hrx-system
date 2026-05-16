// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/channel/bulk/receive_window.h"

#include <cstring>
#include <vector>

#include "iree/base/api.h"
#include "iree/net/channel/bulk/frame.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace net {
namespace {

struct CreditLog {
  std::vector<uint32_t> deltas;
  iree_status_code_t next_status = IREE_STATUS_OK;
};

static iree_status_t RecordCredit(void* user_data, uint32_t credit_delta) {
  auto* log = static_cast<CreditLog*>(user_data);
  if (log->next_status != IREE_STATUS_OK) {
    iree_status_code_t status = log->next_status;
    log->next_status = IREE_STATUS_OK;
    return iree_status_from_code(status);
  }
  log->deltas.push_back(credit_delta);
  return iree_ok_status();
}

struct LeaseReleaseLog {
  std::vector<iree_async_buffer_index_t> released;
};

static void RecordLeaseRelease(void* user_data,
                               iree_async_buffer_index_t buffer_index) {
  auto* log = static_cast<LeaseReleaseLog*>(user_data);
  log->released.push_back(buffer_index);
}

static iree_async_buffer_lease_t MakeLease(LeaseReleaseLog* log,
                                           iree_async_buffer_index_t index) {
  iree_async_buffer_lease_t lease;
  memset(&lease, 0, sizeof(lease));
  lease.release.fn = RecordLeaseRelease;
  lease.release.user_data = log;
  lease.buffer_index = index;
  return lease;
}

static iree_net_bulk_receive_window_callbacks_t MakeCallbacks(
    CreditLog* credit_log) {
  iree_net_bulk_receive_window_callbacks_t callbacks;
  memset(&callbacks, 0, sizeof(callbacks));
  callbacks.send_credit = RecordCredit;
  callbacks.user_data = credit_log;
  return callbacks;
}

class ReceiveWindowTest : public ::testing::Test {
 protected:
  void TearDown() override {
    iree_net_bulk_receive_window_free(window_);
    window_ = nullptr;
  }

  void AllocateWindow(iree_host_size_t capacity = 4,
                      uint32_t credit_batch_threshold = 1) {
    iree_net_bulk_receive_window_options_t options =
        iree_net_bulk_receive_window_options_default();
    options.chunk_pool.capacity = capacity;
    options.credit_batch_threshold = credit_batch_threshold;

    iree_net_bulk_receive_window_callbacks_t callbacks =
        MakeCallbacks(&credit_log_);
    IREE_ASSERT_OK(iree_net_bulk_receive_window_allocate(
        &options, callbacks, iree_allocator_system(), &window_));
  }

  CreditLog credit_log_;
  iree_net_bulk_receive_window_t* window_ = nullptr;
};

TEST_F(ReceiveWindowTest, AllocateDefaultOptions) {
  iree_net_bulk_receive_window_callbacks_t callbacks =
      MakeCallbacks(&credit_log_);
  IREE_ASSERT_OK(iree_net_bulk_receive_window_allocate(
      nullptr, callbacks, iree_allocator_system(), &window_));

  EXPECT_EQ(iree_net_bulk_receive_window_capacity(window_),
            IREE_NET_BULK_CHUNK_POOL_DEFAULT_CAPACITY);
  EXPECT_EQ(iree_net_bulk_receive_window_count(window_), 0u);
  EXPECT_EQ(iree_net_bulk_receive_window_advertised_credit_count(window_), 0u);
  EXPECT_EQ(iree_net_bulk_receive_window_unadvertised_credit_count(window_),
            IREE_NET_BULK_CHUNK_POOL_DEFAULT_CAPACITY);
  EXPECT_TRUE(iree_net_bulk_receive_window_should_flush_credit(window_));
}

TEST_F(ReceiveWindowTest, FlushInitialCreditAdvertisesCapacity) {
  AllocateWindow(/*capacity=*/3);

  IREE_ASSERT_OK(iree_net_bulk_receive_window_flush_credit(window_));

  EXPECT_EQ(credit_log_.deltas, (std::vector<uint32_t>{3}));
  EXPECT_EQ(iree_net_bulk_receive_window_advertised_credit_count(window_), 3u);
  EXPECT_EQ(iree_net_bulk_receive_window_unadvertised_credit_count(window_),
            0u);
  EXPECT_FALSE(iree_net_bulk_receive_window_should_flush_credit(window_));
}

TEST_F(ReceiveWindowTest, FlushCreditBackpressureKeepsPending) {
  AllocateWindow(/*capacity=*/2);

  credit_log_.next_status = IREE_STATUS_RESOURCE_EXHAUSTED;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        iree_net_bulk_receive_window_flush_credit(window_));

  EXPECT_TRUE(credit_log_.deltas.empty());
  EXPECT_EQ(iree_net_bulk_receive_window_advertised_credit_count(window_), 0u);
  EXPECT_EQ(iree_net_bulk_receive_window_unadvertised_credit_count(window_),
            2u);

  IREE_ASSERT_OK(iree_net_bulk_receive_window_flush_credit(window_));
  EXPECT_EQ(credit_log_.deltas, (std::vector<uint32_t>{2}));
  EXPECT_EQ(iree_net_bulk_receive_window_advertised_credit_count(window_), 2u);
  EXPECT_EQ(iree_net_bulk_receive_window_unadvertised_credit_count(window_),
            0u);
}

TEST_F(ReceiveWindowTest, AcquireRequiresAdvertisedCredit) {
  AllocateWindow(/*capacity=*/1);

  uint8_t payload_bytes[] = {0x01};
  iree_const_byte_span_t payload =
      iree_make_const_byte_span(payload_bytes, sizeof(payload_bytes));
  LeaseReleaseLog release_log;
  iree_async_buffer_lease_t lease = MakeLease(&release_log, 1);
  iree_net_bulk_chunk_t* chunk = nullptr;

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_net_bulk_receive_window_acquire_chunk(
                            window_, /*transfer_id=*/1, /*chunk_offset=*/0,
                            /*sequence=*/0, IREE_NET_BULK_FRAME_FLAG_NONE,
                            payload, &lease, /*user_value=*/0, &chunk));
  EXPECT_NE(lease.release.fn, nullptr);

  iree_async_buffer_lease_release(&lease);
  EXPECT_EQ(release_log.released, (std::vector<iree_async_buffer_index_t>{1}));
}

TEST_F(ReceiveWindowTest, AcquireConsumesCreditAndReleaseQueuesCredit) {
  AllocateWindow(/*capacity=*/2);
  IREE_ASSERT_OK(iree_net_bulk_receive_window_flush_credit(window_));

  uint8_t payload_bytes[] = {0x10, 0x11};
  iree_const_byte_span_t payload =
      iree_make_const_byte_span(payload_bytes, sizeof(payload_bytes));
  LeaseReleaseLog release_log;
  iree_async_buffer_lease_t lease = MakeLease(&release_log, 7);

  iree_net_bulk_chunk_t* chunk = nullptr;
  IREE_ASSERT_OK(iree_net_bulk_receive_window_acquire_chunk(
      window_, /*transfer_id=*/42, /*chunk_offset=*/64, /*sequence=*/3,
      IREE_NET_BULK_FRAME_FLAG_FINAL_CHUNK, payload, &lease,
      /*user_value=*/99, &chunk));

  EXPECT_EQ(iree_net_bulk_receive_window_count(window_), 1u);
  EXPECT_EQ(iree_net_bulk_receive_window_advertised_credit_count(window_), 1u);
  EXPECT_EQ(iree_net_bulk_receive_window_unadvertised_credit_count(window_),
            0u);
  EXPECT_EQ(iree_net_bulk_chunk_transfer_id(chunk), 42u);
  EXPECT_EQ(iree_net_bulk_chunk_offset(chunk), 64u);
  EXPECT_EQ(iree_net_bulk_chunk_sequence(chunk), 3u);
  EXPECT_EQ(iree_net_bulk_chunk_flags(chunk),
            IREE_NET_BULK_FRAME_FLAG_FINAL_CHUNK);
  EXPECT_EQ(iree_net_bulk_chunk_user_value(chunk), 99u);
  EXPECT_EQ(lease.release.fn, nullptr);

  iree_net_bulk_receive_window_release_chunk(window_, chunk);
  EXPECT_EQ(iree_net_bulk_receive_window_count(window_), 0u);
  EXPECT_EQ(iree_net_bulk_receive_window_advertised_credit_count(window_), 1u);
  EXPECT_EQ(iree_net_bulk_receive_window_unadvertised_credit_count(window_),
            1u);
  EXPECT_TRUE(iree_net_bulk_receive_window_should_flush_credit(window_));
  EXPECT_EQ(release_log.released, (std::vector<iree_async_buffer_index_t>{7}));

  IREE_ASSERT_OK(iree_net_bulk_receive_window_flush_credit(window_));
  EXPECT_EQ(credit_log_.deltas, (std::vector<uint32_t>{2, 1}));
  EXPECT_EQ(iree_net_bulk_receive_window_advertised_credit_count(window_), 2u);
  EXPECT_EQ(iree_net_bulk_receive_window_unadvertised_credit_count(window_),
            0u);
}

TEST_F(ReceiveWindowTest, DataBeyondCreditDoesNotStealLease) {
  AllocateWindow(/*capacity=*/1);
  IREE_ASSERT_OK(iree_net_bulk_receive_window_flush_credit(window_));

  uint8_t payload_bytes[] = {0x01};
  iree_const_byte_span_t payload =
      iree_make_const_byte_span(payload_bytes, sizeof(payload_bytes));
  LeaseReleaseLog release_log;
  iree_async_buffer_lease_t first_lease = MakeLease(&release_log, 1);
  iree_async_buffer_lease_t second_lease = MakeLease(&release_log, 2);

  iree_net_bulk_chunk_t* first_chunk = nullptr;
  IREE_ASSERT_OK(iree_net_bulk_receive_window_acquire_chunk(
      window_, /*transfer_id=*/1, /*chunk_offset=*/0, /*sequence=*/0,
      IREE_NET_BULK_FRAME_FLAG_NONE, payload, &first_lease,
      /*user_value=*/0, &first_chunk));

  iree_net_bulk_chunk_t* second_chunk = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_bulk_receive_window_acquire_chunk(
          window_, /*transfer_id=*/1, /*chunk_offset=*/1,
          /*sequence=*/1, IREE_NET_BULK_FRAME_FLAG_NONE, payload, &second_lease,
          /*user_value=*/0, &second_chunk));
  EXPECT_NE(second_lease.release.fn, nullptr);
  EXPECT_EQ(iree_net_bulk_receive_window_advertised_credit_count(window_), 0u);
  EXPECT_EQ(iree_net_bulk_receive_window_count(window_), 1u);

  iree_net_bulk_receive_window_release_chunk(window_, first_chunk);
  iree_async_buffer_lease_release(&second_lease);
  EXPECT_EQ(release_log.released,
            (std::vector<iree_async_buffer_index_t>{1, 2}));
}

TEST_F(ReceiveWindowTest, MalformedPayloadDoesNotConsumeCredit) {
  AllocateWindow(/*capacity=*/1);
  IREE_ASSERT_OK(iree_net_bulk_receive_window_flush_credit(window_));

  uint8_t payload_bytes[] = {0x01};
  iree_const_byte_span_t payload =
      iree_make_const_byte_span(payload_bytes, sizeof(payload_bytes));
  iree_net_bulk_chunk_t* chunk = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_bulk_receive_window_acquire_chunk(
          window_, /*transfer_id=*/1, /*chunk_offset=*/0,
          /*sequence=*/0, IREE_NET_BULK_FRAME_FLAG_NONE, payload,
          /*lease=*/nullptr, /*user_value=*/0, &chunk));

  EXPECT_EQ(iree_net_bulk_receive_window_advertised_credit_count(window_), 1u);
  EXPECT_EQ(iree_net_bulk_receive_window_unadvertised_credit_count(window_),
            0u);
  EXPECT_EQ(iree_net_bulk_receive_window_count(window_), 0u);
}

TEST_F(ReceiveWindowTest, CreditBatchThresholdIsAFlushHint) {
  AllocateWindow(/*capacity=*/2, /*credit_batch_threshold=*/2);
  IREE_ASSERT_OK(iree_net_bulk_receive_window_flush_credit(window_));

  uint8_t payload_bytes[] = {0x01};
  iree_const_byte_span_t payload =
      iree_make_const_byte_span(payload_bytes, sizeof(payload_bytes));
  LeaseReleaseLog release_log;
  iree_async_buffer_lease_t first_lease = MakeLease(&release_log, 1);
  iree_async_buffer_lease_t second_lease = MakeLease(&release_log, 2);

  iree_net_bulk_chunk_t* first_chunk = nullptr;
  IREE_ASSERT_OK(iree_net_bulk_receive_window_acquire_chunk(
      window_, /*transfer_id=*/1, /*chunk_offset=*/0, /*sequence=*/0,
      IREE_NET_BULK_FRAME_FLAG_NONE, payload, &first_lease,
      /*user_value=*/0, &first_chunk));
  iree_net_bulk_chunk_t* second_chunk = nullptr;
  IREE_ASSERT_OK(iree_net_bulk_receive_window_acquire_chunk(
      window_, /*transfer_id=*/1, /*chunk_offset=*/1, /*sequence=*/1,
      IREE_NET_BULK_FRAME_FLAG_NONE, payload, &second_lease,
      /*user_value=*/0, &second_chunk));

  iree_net_bulk_receive_window_release_chunk(window_, first_chunk);
  EXPECT_FALSE(iree_net_bulk_receive_window_should_flush_credit(window_));
  iree_net_bulk_receive_window_release_chunk(window_, second_chunk);
  EXPECT_TRUE(iree_net_bulk_receive_window_should_flush_credit(window_));

  IREE_ASSERT_OK(iree_net_bulk_receive_window_flush_credit(window_));
  EXPECT_EQ(credit_log_.deltas, (std::vector<uint32_t>{2, 2}));
}

TEST_F(ReceiveWindowTest, CreditBatchThresholdClampsToCapacity) {
  AllocateWindow(/*capacity=*/2, /*credit_batch_threshold=*/16);
  EXPECT_TRUE(iree_net_bulk_receive_window_should_flush_credit(window_));
}

TEST_F(ReceiveWindowTest, RejectsMissingCreditCallback) {
  iree_net_bulk_receive_window_callbacks_t callbacks = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_bulk_receive_window_allocate(nullptr, callbacks,
                                            iree_allocator_system(), &window_));
}

TEST_F(ReceiveWindowTest, RejectsCapacityTooLargeForCreditFrame) {
  iree_net_bulk_receive_window_options_t options =
      iree_net_bulk_receive_window_options_default();
  options.chunk_pool.capacity = (iree_host_size_t)INT32_MAX + 1;
  iree_net_bulk_receive_window_callbacks_t callbacks =
      MakeCallbacks(&credit_log_);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_net_bulk_receive_window_allocate(&options, callbacks,
                                            iree_allocator_system(), &window_));
}

}  // namespace
}  // namespace net
}  // namespace iree
