// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/util/bulk_transfer_scheduler.h"

#include <algorithm>
#include <cstring>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

typedef struct test_transfer_storage_t {
  // Test-owned transfer kind marker.
  uint32_t kind;

  // Test-owned payload value.
  uint32_t value;
} test_transfer_storage_t;

struct DeinitializeLog {
  // Transfer IDs observed by the deinitializer callback.
  std::vector<uint64_t> transfer_ids;
};

static void RecordDeinitialize(void* user_data,
                               iree_net_bulk_transfer_t* transfer) {
  auto* log = static_cast<DeinitializeLog*>(user_data);
  log->transfer_ids.push_back(iree_net_bulk_transfer_id(transfer));
  auto storage = reinterpret_cast<test_transfer_storage_t*>(
      iree_net_bulk_transfer_user_storage(transfer).data);
  storage->kind = 0;
  storage->value = 0;
}

static bool SelectEvenTransferIds(void* user_data,
                                  iree_net_bulk_transfer_t* transfer) {
  (void)user_data;
  return (iree_net_bulk_transfer_id(transfer) & 1) == 0;
}

class BulkTransferSchedulerTest : public ::testing::Test {
 protected:
  void TearDown() override {
    iree_hal_remote_bulk_transfer_scheduler_free(scheduler_);
    scheduler_ = nullptr;
  }

  void AllocateScheduler(iree_host_size_t capacity = 4,
                         DeinitializeLog* log = nullptr) {
    iree_hal_remote_bulk_transfer_scheduler_options_t options =
        iree_hal_remote_bulk_transfer_scheduler_options_default();
    options.capacity = capacity;
    options.user_storage_size = sizeof(test_transfer_storage_t);
    options.user_storage_alignment = iree_alignof(test_transfer_storage_t);
    options.initial_transfer_id = 2;
    options.transfer_id_stride = 2;
    iree_hal_remote_bulk_transfer_scheduler_callbacks_t callbacks = {};
    callbacks.deinitialize = log ? RecordDeinitialize : nullptr;
    callbacks.user_data = log;
    IREE_ASSERT_OK(iree_hal_remote_bulk_transfer_scheduler_allocate(
        &options, callbacks, iree_allocator_system(), &scheduler_));
  }

  // Scheduler under test.
  iree_hal_remote_bulk_transfer_scheduler_t* scheduler_ = nullptr;
};

TEST_F(BulkTransferSchedulerTest, AllocateDefaults) {
  IREE_ASSERT_OK(iree_hal_remote_bulk_transfer_scheduler_allocate(
      nullptr, iree_hal_remote_bulk_transfer_scheduler_callbacks_t{},
      iree_allocator_system(), &scheduler_));

  EXPECT_EQ(iree_hal_remote_bulk_transfer_scheduler_capacity(scheduler_),
            IREE_NET_BULK_TRANSFER_TABLE_DEFAULT_CAPACITY);
  EXPECT_EQ(iree_hal_remote_bulk_transfer_scheduler_count(scheduler_), 0u);
  EXPECT_TRUE(iree_hal_remote_bulk_transfer_scheduler_has_capacity(scheduler_));
}

TEST_F(BulkTransferSchedulerTest, InsertLookupAndReleasePeerTransfer) {
  DeinitializeLog log;
  AllocateScheduler(/*capacity=*/2, &log);

  iree_net_bulk_transfer_t* transfer = nullptr;
  IREE_ASSERT_OK(iree_hal_remote_bulk_transfer_scheduler_insert_peer(
      scheduler_, /*transfer_id=*/42, /*total_size=*/4096,
      /*user_value=*/7, &transfer));
  EXPECT_EQ(iree_hal_remote_bulk_transfer_scheduler_count(scheduler_), 1u);
  EXPECT_EQ(iree_hal_remote_bulk_transfer_scheduler_lookup(scheduler_, 42),
            transfer);

  auto storage = reinterpret_cast<test_transfer_storage_t*>(
      iree_net_bulk_transfer_user_storage(transfer).data);
  storage->kind = 1;
  storage->value = 99;

  EXPECT_TRUE(
      iree_hal_remote_bulk_transfer_scheduler_release(scheduler_, transfer));
  EXPECT_EQ(iree_hal_remote_bulk_transfer_scheduler_count(scheduler_), 0u);
  EXPECT_EQ(log.transfer_ids, (std::vector<uint64_t>{42}));
  EXPECT_EQ(iree_hal_remote_bulk_transfer_scheduler_lookup(scheduler_, 42),
            nullptr);
}

TEST_F(BulkTransferSchedulerTest, AllocateLocalUsesConfiguredIdSequence) {
  AllocateScheduler(/*capacity=*/2);

  iree_net_bulk_transfer_t* first = nullptr;
  iree_net_bulk_transfer_t* second = nullptr;
  IREE_ASSERT_OK(iree_hal_remote_bulk_transfer_scheduler_allocate_local(
      scheduler_, /*total_size=*/16, /*user_value=*/1, &first));
  IREE_ASSERT_OK(iree_hal_remote_bulk_transfer_scheduler_allocate_local(
      scheduler_, /*total_size=*/32, /*user_value=*/2, &second));

  EXPECT_EQ(iree_net_bulk_transfer_id(first), 2u);
  EXPECT_EQ(iree_net_bulk_transfer_id(second), 4u);
  EXPECT_FALSE(
      iree_hal_remote_bulk_transfer_scheduler_has_capacity(scheduler_));
}

TEST_F(BulkTransferSchedulerTest, CapacityExhaustionDoesNotDeinitialize) {
  DeinitializeLog log;
  AllocateScheduler(/*capacity=*/1, &log);

  iree_net_bulk_transfer_t* transfer = nullptr;
  IREE_ASSERT_OK(iree_hal_remote_bulk_transfer_scheduler_insert_peer(
      scheduler_, /*transfer_id=*/1, /*total_size=*/1,
      /*user_value=*/0, &transfer));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        iree_hal_remote_bulk_transfer_scheduler_insert_peer(
                            scheduler_, /*transfer_id=*/2, /*total_size=*/1,
                            /*user_value=*/0, &transfer));
  EXPECT_TRUE(log.transfer_ids.empty());

  iree_hal_remote_bulk_transfer_scheduler_clear(scheduler_);
  EXPECT_EQ(log.transfer_ids, (std::vector<uint64_t>{1}));
}

TEST_F(BulkTransferSchedulerTest, ClearDeinitializesActiveTransfers) {
  DeinitializeLog log;
  AllocateScheduler(/*capacity=*/4, &log);

  iree_net_bulk_transfer_t* transfer = nullptr;
  IREE_ASSERT_OK(iree_hal_remote_bulk_transfer_scheduler_insert_peer(
      scheduler_, /*transfer_id=*/10, /*total_size=*/1,
      /*user_value=*/0, &transfer));
  IREE_ASSERT_OK(iree_hal_remote_bulk_transfer_scheduler_insert_peer(
      scheduler_, /*transfer_id=*/20, /*total_size=*/1,
      /*user_value=*/0, &transfer));

  iree_hal_remote_bulk_transfer_scheduler_clear(scheduler_);
  std::sort(log.transfer_ids.begin(), log.transfer_ids.end());
  EXPECT_EQ(log.transfer_ids, (std::vector<uint64_t>{10, 20}));
  EXPECT_EQ(iree_hal_remote_bulk_transfer_scheduler_count(scheduler_), 0u);
}

TEST_F(BulkTransferSchedulerTest, CollectTransferIdsSupportsTwoPhaseRetry) {
  AllocateScheduler(/*capacity=*/4);

  iree_net_bulk_transfer_t* transfer = nullptr;
  for (uint64_t transfer_id = 1; transfer_id <= 4; ++transfer_id) {
    IREE_ASSERT_OK(iree_hal_remote_bulk_transfer_scheduler_insert_peer(
        scheduler_, transfer_id, /*total_size=*/1, /*user_value=*/0,
        &transfer));
  }

  uint64_t transfer_ids[4] = {0};
  iree_host_size_t transfer_count = 0;
  EXPECT_TRUE(iree_hal_remote_bulk_transfer_scheduler_collect_transfer_ids(
      scheduler_, SelectEvenTransferIds, nullptr, transfer_ids,
      IREE_ARRAYSIZE(transfer_ids), &transfer_count));
  std::sort(transfer_ids, transfer_ids + transfer_count);
  EXPECT_EQ(transfer_count, 2u);
  EXPECT_EQ(transfer_ids[0], 2u);
  EXPECT_EQ(transfer_ids[1], 4u);
}

TEST_F(BulkTransferSchedulerTest, CollectTransferIdsReportsOverflow) {
  AllocateScheduler(/*capacity=*/2);

  iree_net_bulk_transfer_t* transfer = nullptr;
  IREE_ASSERT_OK(iree_hal_remote_bulk_transfer_scheduler_insert_peer(
      scheduler_, /*transfer_id=*/1, /*total_size=*/1,
      /*user_value=*/0, &transfer));
  IREE_ASSERT_OK(iree_hal_remote_bulk_transfer_scheduler_insert_peer(
      scheduler_, /*transfer_id=*/2, /*total_size=*/1,
      /*user_value=*/0, &transfer));

  uint64_t transfer_id = 0;
  iree_host_size_t transfer_count = 0;
  EXPECT_FALSE(iree_hal_remote_bulk_transfer_scheduler_collect_transfer_ids(
      scheduler_, /*select=*/nullptr, nullptr, &transfer_id,
      /*capacity=*/1, &transfer_count));
  EXPECT_EQ(transfer_count, 1u);
}

}  // namespace
