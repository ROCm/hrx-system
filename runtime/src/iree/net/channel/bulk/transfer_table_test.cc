// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/channel/bulk/transfer_table.h"

#include <algorithm>
#include <cstring>
#include <vector>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace net {
namespace {

class TransferTableTest : public ::testing::Test {
 protected:
  void TearDown() override {
    iree_net_bulk_transfer_table_free(table_);
    table_ = nullptr;
  }

  void AllocateTable(iree_host_size_t capacity = 4,
                     iree_host_size_t user_storage_size = 0,
                     iree_host_size_t user_storage_alignment = 0) {
    iree_net_bulk_transfer_table_options_t options =
        iree_net_bulk_transfer_table_options_default();
    options.capacity = capacity;
    options.user_storage_size = user_storage_size;
    options.user_storage_alignment = user_storage_alignment;
    IREE_ASSERT_OK(iree_net_bulk_transfer_table_allocate(
        &options, iree_allocator_system(), &table_));
  }

  iree_net_bulk_transfer_table_t* table_ = nullptr;
};

TEST_F(TransferTableTest, AllocateDefaultOptions) {
  IREE_ASSERT_OK(iree_net_bulk_transfer_table_allocate(
      nullptr, iree_allocator_system(), &table_));
  EXPECT_EQ(iree_net_bulk_transfer_table_capacity(table_),
            IREE_NET_BULK_TRANSFER_TABLE_DEFAULT_CAPACITY);
  EXPECT_EQ(iree_net_bulk_transfer_table_count(table_), 0u);
}

TEST_F(TransferTableTest, InsertLookupAndRemove) {
  AllocateTable();

  iree_net_bulk_transfer_t* transfer = nullptr;
  IREE_ASSERT_OK(iree_net_bulk_transfer_table_insert(
      table_, /*transfer_id=*/42, /*total_size=*/4096,
      /*user_value=*/7, &transfer));
  EXPECT_EQ(iree_net_bulk_transfer_table_count(table_), 1u);
  EXPECT_EQ(iree_net_bulk_transfer_id(transfer), 42u);
  EXPECT_EQ(iree_net_bulk_transfer_total_size(transfer), 4096u);
  EXPECT_EQ(iree_net_bulk_transfer_user_value(transfer), 7u);

  iree_net_bulk_transfer_t* lookup =
      iree_net_bulk_transfer_table_lookup(table_, 42);
  EXPECT_EQ(lookup, transfer);

  iree_net_bulk_transfer_set_user_value(transfer, 99);
  EXPECT_EQ(iree_net_bulk_transfer_user_value(lookup), 99u);

  EXPECT_TRUE(iree_net_bulk_transfer_table_remove(table_, 42));
  EXPECT_EQ(iree_net_bulk_transfer_table_count(table_), 0u);
  EXPECT_EQ(iree_net_bulk_transfer_table_lookup(table_, 42), nullptr);
  EXPECT_FALSE(iree_net_bulk_transfer_table_remove(table_, 42));
}

TEST_F(TransferTableTest, RejectsZeroAndDuplicateTransferIds) {
  AllocateTable();

  iree_net_bulk_transfer_t* transfer = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_net_bulk_transfer_table_insert(
                            table_, /*transfer_id=*/0, /*total_size=*/1,
                            /*user_value=*/0, &transfer));

  IREE_ASSERT_OK(iree_net_bulk_transfer_table_insert(
      table_, /*transfer_id=*/1, /*total_size=*/1, /*user_value=*/0,
      &transfer));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ALREADY_EXISTS,
                        iree_net_bulk_transfer_table_insert(
                            table_, /*transfer_id=*/1, /*total_size=*/1,
                            /*user_value=*/0, &transfer));
}

TEST_F(TransferTableTest, CapacityExhaustionDoesNotAllocateMore) {
  AllocateTable(/*capacity=*/2);

  iree_net_bulk_transfer_t* transfer = nullptr;
  IREE_ASSERT_OK(iree_net_bulk_transfer_table_insert(
      table_, /*transfer_id=*/1, /*total_size=*/1, /*user_value=*/0,
      &transfer));
  IREE_ASSERT_OK(iree_net_bulk_transfer_table_insert(
      table_, /*transfer_id=*/2, /*total_size=*/1, /*user_value=*/0,
      &transfer));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        iree_net_bulk_transfer_table_insert(
                            table_, /*transfer_id=*/3, /*total_size=*/1,
                            /*user_value=*/0, &transfer));

  EXPECT_TRUE(iree_net_bulk_transfer_table_remove(table_, 1));
  IREE_ASSERT_OK(iree_net_bulk_transfer_table_insert(
      table_, /*transfer_id=*/3, /*total_size=*/1, /*user_value=*/0,
      &transfer));
  EXPECT_EQ(iree_net_bulk_transfer_table_count(table_), 2u);
}

TEST_F(TransferTableTest, GeneratedTransferIdsUseConfiguredSequence) {
  iree_net_bulk_transfer_table_options_t options =
      iree_net_bulk_transfer_table_options_default();
  options.capacity = 4;
  options.initial_transfer_id = 11;
  options.transfer_id_stride = 2;
  IREE_ASSERT_OK(iree_net_bulk_transfer_table_allocate(
      &options, iree_allocator_system(), &table_));

  iree_net_bulk_transfer_t* first = nullptr;
  iree_net_bulk_transfer_t* second = nullptr;
  IREE_ASSERT_OK(iree_net_bulk_transfer_table_allocate_transfer(
      table_, /*total_size=*/16, /*user_value=*/1, &first));
  IREE_ASSERT_OK(iree_net_bulk_transfer_table_allocate_transfer(
      table_, /*total_size=*/32, /*user_value=*/2, &second));

  EXPECT_EQ(iree_net_bulk_transfer_id(first), 11u);
  EXPECT_EQ(iree_net_bulk_transfer_id(second), 13u);
  EXPECT_EQ(iree_net_bulk_transfer_total_size(second), 32u);
}

TEST_F(TransferTableTest, GeneratedTransferIdsSkipLiveCollisions) {
  iree_net_bulk_transfer_table_options_t options =
      iree_net_bulk_transfer_table_options_default();
  options.capacity = 4;
  options.initial_transfer_id = 5;
  options.transfer_id_stride = 1;
  IREE_ASSERT_OK(iree_net_bulk_transfer_table_allocate(
      &options, iree_allocator_system(), &table_));

  iree_net_bulk_transfer_t* transfer = nullptr;
  IREE_ASSERT_OK(iree_net_bulk_transfer_table_insert(
      table_, /*transfer_id=*/5, /*total_size=*/1, /*user_value=*/0,
      &transfer));

  iree_net_bulk_transfer_t* generated = nullptr;
  IREE_ASSERT_OK(iree_net_bulk_transfer_table_allocate_transfer(
      table_, /*total_size=*/1, /*user_value=*/0, &generated));
  EXPECT_EQ(iree_net_bulk_transfer_id(generated), 6u);
}

TEST_F(TransferTableTest, DescriptorPointersStayStableAcrossMapDeletion) {
  AllocateTable(/*capacity=*/8);

  std::vector<iree_net_bulk_transfer_t*> transfers;
  for (uint64_t transfer_id = 1; transfer_id <= 6; ++transfer_id) {
    iree_net_bulk_transfer_t* transfer = nullptr;
    IREE_ASSERT_OK(iree_net_bulk_transfer_table_insert(
        table_, transfer_id, /*total_size=*/transfer_id * 10,
        /*user_value=*/transfer_id * 100, &transfer));
    transfers.push_back(transfer);
  }

  EXPECT_TRUE(iree_net_bulk_transfer_table_remove(table_, 3));
  EXPECT_TRUE(iree_net_bulk_transfer_table_remove(table_, 5));

  for (uint64_t transfer_id : {1ull, 2ull, 4ull, 6ull}) {
    iree_net_bulk_transfer_t* lookup =
        iree_net_bulk_transfer_table_lookup(table_, transfer_id);
    ASSERT_NE(lookup, nullptr);
    EXPECT_EQ(lookup, transfers[transfer_id - 1]);
    EXPECT_EQ(iree_net_bulk_transfer_user_value(lookup), transfer_id * 100);
  }
}

static void RecordVisitedTransfer(void* user_data,
                                  iree_net_bulk_transfer_t* transfer) {
  auto* visited = static_cast<std::vector<uint64_t>*>(user_data);
  visited->push_back(iree_net_bulk_transfer_id(transfer));
}

TEST_F(TransferTableTest, VisitAndClearActiveTransfers) {
  AllocateTable(/*capacity=*/4);

  iree_net_bulk_transfer_t* transfer = nullptr;
  IREE_ASSERT_OK(iree_net_bulk_transfer_table_insert(
      table_, /*transfer_id=*/10, /*total_size=*/1, /*user_value=*/0,
      &transfer));
  IREE_ASSERT_OK(iree_net_bulk_transfer_table_insert(
      table_, /*transfer_id=*/20, /*total_size=*/1, /*user_value=*/0,
      &transfer));

  std::vector<uint64_t> visited;
  iree_net_bulk_transfer_table_visit(table_, RecordVisitedTransfer, &visited);
  std::sort(visited.begin(), visited.end());
  EXPECT_EQ(visited, (std::vector<uint64_t>{10, 20}));

  iree_net_bulk_transfer_table_clear(table_);
  EXPECT_EQ(iree_net_bulk_transfer_table_count(table_), 0u);
  EXPECT_EQ(iree_net_bulk_transfer_table_lookup(table_, 10), nullptr);
  EXPECT_EQ(iree_net_bulk_transfer_table_lookup(table_, 20), nullptr);

  IREE_ASSERT_OK(iree_net_bulk_transfer_table_insert(
      table_, /*transfer_id=*/30, /*total_size=*/1, /*user_value=*/0,
      &transfer));
  EXPECT_NE(iree_net_bulk_transfer_table_lookup(table_, 30), nullptr);
}

TEST_F(TransferTableTest, UserStorageIsStableAlignedAndClearedOnReuse) {
  AllocateTable(/*capacity=*/2, /*user_storage_size=*/24,
                /*user_storage_alignment=*/16);

  iree_net_bulk_transfer_t* transfer = nullptr;
  IREE_ASSERT_OK(iree_net_bulk_transfer_table_insert(
      table_, /*transfer_id=*/1, /*total_size=*/1, /*user_value=*/0,
      &transfer));
  iree_byte_span_t storage = iree_net_bulk_transfer_user_storage(transfer);
  ASSERT_EQ(storage.data_length, 24u);
  EXPECT_TRUE(iree_host_ptr_has_alignment(storage.data, 16));

  memset(storage.data, 0xA5, storage.data_length);
  EXPECT_TRUE(iree_net_bulk_transfer_table_remove(table_, 1));

  IREE_ASSERT_OK(iree_net_bulk_transfer_table_insert(
      table_, /*transfer_id=*/2, /*total_size=*/1, /*user_value=*/0,
      &transfer));
  storage = iree_net_bulk_transfer_user_storage(transfer);
  for (iree_host_size_t i = 0; i < storage.data_length; ++i) {
    EXPECT_EQ(storage.data[i], 0u);
  }
}

TEST_F(TransferTableTest, RejectsInvalidAlignment) {
  iree_net_bulk_transfer_table_options_t options =
      iree_net_bulk_transfer_table_options_default();
  options.capacity = 4;
  options.user_storage_size = 8;
  options.user_storage_alignment = 3;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_net_bulk_transfer_table_allocate(
                            &options, iree_allocator_system(), &table_));
}

}  // namespace
}  // namespace net
}  // namespace iree
