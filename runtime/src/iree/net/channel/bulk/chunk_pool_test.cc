// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/channel/bulk/chunk_pool.h"

#include <algorithm>
#include <cstring>
#include <vector>

#include "iree/base/api.h"
#include "iree/net/channel/bulk/frame.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace net {
namespace {

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

class ChunkPoolTest : public ::testing::Test {
 protected:
  void TearDown() override {
    iree_net_bulk_chunk_pool_free(pool_);
    pool_ = nullptr;
  }

  void AllocatePool(iree_host_size_t capacity = 4,
                    iree_host_size_t user_storage_size = 0,
                    iree_host_size_t user_storage_alignment = 0) {
    iree_net_bulk_chunk_pool_options_t options =
        iree_net_bulk_chunk_pool_options_default();
    options.capacity = capacity;
    options.user_storage_size = user_storage_size;
    options.user_storage_alignment = user_storage_alignment;
    IREE_ASSERT_OK(iree_net_bulk_chunk_pool_allocate(
        &options, iree_allocator_system(), &pool_));
  }

  iree_net_bulk_chunk_pool_t* pool_ = nullptr;
};

TEST_F(ChunkPoolTest, AllocateDefaultOptions) {
  IREE_ASSERT_OK(iree_net_bulk_chunk_pool_allocate(
      nullptr, iree_allocator_system(), &pool_));
  EXPECT_EQ(iree_net_bulk_chunk_pool_capacity(pool_),
            IREE_NET_BULK_CHUNK_POOL_DEFAULT_CAPACITY);
  EXPECT_EQ(iree_net_bulk_chunk_pool_count(pool_), 0u);
}

TEST_F(ChunkPoolTest, AcquireStealsLeaseAndReleaseReturnsIt) {
  AllocatePool();

  uint8_t payload_bytes[] = {0x10, 0x11, 0x12};
  iree_const_byte_span_t payload =
      iree_make_const_byte_span(payload_bytes, sizeof(payload_bytes));
  LeaseReleaseLog release_log;
  iree_async_buffer_lease_t lease = MakeLease(&release_log, 7);

  iree_net_bulk_chunk_t* chunk = nullptr;
  IREE_ASSERT_OK(iree_net_bulk_chunk_pool_acquire(
      pool_, /*transfer_id=*/42, /*chunk_offset=*/128, /*sequence=*/9,
      IREE_NET_BULK_FRAME_FLAG_FINAL_CHUNK, payload, &lease,
      /*user_value=*/99, &chunk));

  EXPECT_EQ(iree_net_bulk_chunk_pool_count(pool_), 1u);
  EXPECT_EQ(iree_net_bulk_chunk_transfer_id(chunk), 42u);
  EXPECT_EQ(iree_net_bulk_chunk_offset(chunk), 128u);
  EXPECT_EQ(iree_net_bulk_chunk_sequence(chunk), 9u);
  EXPECT_EQ(iree_net_bulk_chunk_flags(chunk),
            IREE_NET_BULK_FRAME_FLAG_FINAL_CHUNK);
  EXPECT_EQ(iree_net_bulk_chunk_user_value(chunk), 99u);
  EXPECT_EQ(iree_net_bulk_chunk_payload(chunk).data, payload.data);
  EXPECT_EQ(iree_net_bulk_chunk_payload(chunk).data_length,
            payload.data_length);
  EXPECT_EQ(lease.release.fn, nullptr);

  iree_net_bulk_chunk_set_user_value(chunk, 123);
  EXPECT_EQ(iree_net_bulk_chunk_user_value(chunk), 123u);

  iree_net_bulk_chunk_release(pool_, chunk);
  EXPECT_EQ(iree_net_bulk_chunk_pool_count(pool_), 0u);
  EXPECT_EQ(release_log.released, (std::vector<iree_async_buffer_index_t>{7}));
}

TEST_F(ChunkPoolTest, RejectsNonEmptyPayloadWithoutRetainableLease) {
  AllocatePool();

  uint8_t payload_bytes[] = {0x01};
  iree_const_byte_span_t payload =
      iree_make_const_byte_span(payload_bytes, sizeof(payload_bytes));

  iree_net_bulk_chunk_t* chunk = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_bulk_chunk_pool_acquire(
          pool_, /*transfer_id=*/1, /*chunk_offset=*/0,
          /*sequence=*/0, IREE_NET_BULK_FRAME_FLAG_NONE, payload,
          /*lease=*/nullptr, /*user_value=*/0, &chunk));

  iree_async_buffer_lease_t lease;
  memset(&lease, 0, sizeof(lease));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_net_bulk_chunk_pool_acquire(
                            pool_, /*transfer_id=*/1, /*chunk_offset=*/0,
                            /*sequence=*/0, IREE_NET_BULK_FRAME_FLAG_NONE,
                            payload, &lease, /*user_value=*/0, &chunk));
}

TEST_F(ChunkPoolTest, RejectsInvalidAcquireArguments) {
  AllocatePool();

  iree_net_bulk_chunk_t* chunk = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_net_bulk_chunk_pool_acquire(
                            pool_, /*transfer_id=*/0, /*chunk_offset=*/0,
                            /*sequence=*/0, IREE_NET_BULK_FRAME_FLAG_NONE,
                            iree_const_byte_span_empty(), /*lease=*/nullptr,
                            /*user_value=*/0, &chunk));

  iree_const_byte_span_t payload = {nullptr, 1};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_bulk_chunk_pool_acquire(
          pool_, /*transfer_id=*/1, /*chunk_offset=*/0,
          /*sequence=*/0, IREE_NET_BULK_FRAME_FLAG_NONE, payload,
          /*lease=*/nullptr, /*user_value=*/0, &chunk));
}

TEST_F(ChunkPoolTest, AllowsEmptyPayloadWithoutLease) {
  AllocatePool();

  iree_net_bulk_chunk_t* chunk = nullptr;
  IREE_ASSERT_OK(iree_net_bulk_chunk_pool_acquire(
      pool_, /*transfer_id=*/1, /*chunk_offset=*/0, /*sequence=*/0,
      IREE_NET_BULK_FRAME_FLAG_NONE, iree_const_byte_span_empty(),
      /*lease=*/nullptr, /*user_value=*/0, &chunk));
  EXPECT_EQ(iree_net_bulk_chunk_payload(chunk).data_length, 0u);
  iree_net_bulk_chunk_release(pool_, chunk);
}

TEST_F(ChunkPoolTest, EmptyPayloadDoesNotStealLease) {
  AllocatePool();

  LeaseReleaseLog release_log;
  iree_async_buffer_lease_t lease = MakeLease(&release_log, 7);

  iree_net_bulk_chunk_t* chunk = nullptr;
  IREE_ASSERT_OK(iree_net_bulk_chunk_pool_acquire(
      pool_, /*transfer_id=*/1, /*chunk_offset=*/0, /*sequence=*/0,
      IREE_NET_BULK_FRAME_FLAG_NONE, iree_const_byte_span_empty(), &lease,
      /*user_value=*/0, &chunk));
  EXPECT_NE(lease.release.fn, nullptr);

  iree_net_bulk_chunk_release(pool_, chunk);
  EXPECT_TRUE(release_log.released.empty());

  iree_async_buffer_lease_release(&lease);
  EXPECT_EQ(release_log.released, (std::vector<iree_async_buffer_index_t>{7}));
}

TEST_F(ChunkPoolTest, CapacityExhaustionDoesNotStealLease) {
  AllocatePool(/*capacity=*/1);

  uint8_t payload_bytes[] = {0x01};
  iree_const_byte_span_t payload =
      iree_make_const_byte_span(payload_bytes, sizeof(payload_bytes));
  LeaseReleaseLog release_log;
  iree_async_buffer_lease_t first_lease = MakeLease(&release_log, 1);
  iree_async_buffer_lease_t second_lease = MakeLease(&release_log, 2);

  iree_net_bulk_chunk_t* first_chunk = nullptr;
  IREE_ASSERT_OK(iree_net_bulk_chunk_pool_acquire(
      pool_, /*transfer_id=*/1, /*chunk_offset=*/0, /*sequence=*/0,
      IREE_NET_BULK_FRAME_FLAG_NONE, payload, &first_lease,
      /*user_value=*/0, &first_chunk));
  iree_net_bulk_chunk_t* chunk = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        iree_net_bulk_chunk_pool_acquire(
                            pool_, /*transfer_id=*/2, /*chunk_offset=*/0,
                            /*sequence=*/0, IREE_NET_BULK_FRAME_FLAG_NONE,
                            payload, &second_lease, /*user_value=*/0, &chunk));
  EXPECT_NE(second_lease.release.fn, nullptr);

  iree_net_bulk_chunk_release(pool_, first_chunk);
  iree_async_buffer_lease_release(&second_lease);
  EXPECT_EQ(release_log.released,
            (std::vector<iree_async_buffer_index_t>{1, 2}));
}

static void RecordVisitedChunk(void* user_data, iree_net_bulk_chunk_t* chunk) {
  auto* visited = static_cast<std::vector<uint64_t>*>(user_data);
  visited->push_back(iree_net_bulk_chunk_transfer_id(chunk));
}

TEST_F(ChunkPoolTest, VisitAndClearReleaseActiveLeases) {
  AllocatePool(/*capacity=*/3);

  uint8_t payload_bytes[] = {0x01};
  iree_const_byte_span_t payload =
      iree_make_const_byte_span(payload_bytes, sizeof(payload_bytes));
  LeaseReleaseLog release_log;
  iree_async_buffer_lease_t first_lease = MakeLease(&release_log, 10);
  iree_async_buffer_lease_t second_lease = MakeLease(&release_log, 20);

  iree_net_bulk_chunk_t* chunk = nullptr;
  IREE_ASSERT_OK(iree_net_bulk_chunk_pool_acquire(
      pool_, /*transfer_id=*/100, /*chunk_offset=*/0, /*sequence=*/0,
      IREE_NET_BULK_FRAME_FLAG_NONE, payload, &first_lease,
      /*user_value=*/0, &chunk));
  IREE_ASSERT_OK(iree_net_bulk_chunk_pool_acquire(
      pool_, /*transfer_id=*/200, /*chunk_offset=*/0, /*sequence=*/0,
      IREE_NET_BULK_FRAME_FLAG_NONE, payload, &second_lease,
      /*user_value=*/0, &chunk));

  std::vector<uint64_t> visited;
  iree_net_bulk_chunk_pool_visit(pool_, RecordVisitedChunk, &visited);
  std::sort(visited.begin(), visited.end());
  EXPECT_EQ(visited, (std::vector<uint64_t>{100, 200}));

  iree_net_bulk_chunk_pool_clear(pool_);
  EXPECT_EQ(iree_net_bulk_chunk_pool_count(pool_), 0u);
  EXPECT_EQ(release_log.released,
            (std::vector<iree_async_buffer_index_t>{10, 20}));
}

TEST_F(ChunkPoolTest, UserStorageIsStableAlignedAndClearedOnReuse) {
  AllocatePool(/*capacity=*/1, /*user_storage_size=*/24,
               /*user_storage_alignment=*/16);

  iree_net_bulk_chunk_t* chunk = nullptr;
  IREE_ASSERT_OK(iree_net_bulk_chunk_pool_acquire(
      pool_, /*transfer_id=*/1, /*chunk_offset=*/0, /*sequence=*/0,
      IREE_NET_BULK_FRAME_FLAG_NONE, iree_const_byte_span_empty(),
      /*lease=*/nullptr, /*user_value=*/0, &chunk));
  iree_byte_span_t storage = iree_net_bulk_chunk_user_storage(chunk);
  ASSERT_EQ(storage.data_length, 24u);
  EXPECT_TRUE(iree_host_ptr_has_alignment(storage.data, 16));

  memset(storage.data, 0xA5, storage.data_length);
  iree_net_bulk_chunk_release(pool_, chunk);

  IREE_ASSERT_OK(iree_net_bulk_chunk_pool_acquire(
      pool_, /*transfer_id=*/2, /*chunk_offset=*/0, /*sequence=*/0,
      IREE_NET_BULK_FRAME_FLAG_NONE, iree_const_byte_span_empty(),
      /*lease=*/nullptr, /*user_value=*/0, &chunk));
  storage = iree_net_bulk_chunk_user_storage(chunk);
  for (iree_host_size_t i = 0; i < storage.data_length; ++i) {
    EXPECT_EQ(storage.data[i], 0u);
  }
}

TEST_F(ChunkPoolTest, RejectsInvalidOptions) {
  iree_net_bulk_chunk_pool_options_t options =
      iree_net_bulk_chunk_pool_options_default();
  options.capacity = 4;
  options.user_storage_size = 8;
  options.user_storage_alignment = 3;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_net_bulk_chunk_pool_allocate(
                            &options, iree_allocator_system(), &pool_));
}

}  // namespace
}  // namespace net
}  // namespace iree
