// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/util/operation_completion.h"

#include "iree/async/operation.h"
#include "iree/async/util/operation_pool.h"
#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

struct CompletionRecord {
  iree_host_size_t call_count = 0;
  iree_status_code_t status_code = IREE_STATUS_OK;
  iree_async_completion_flags_t flags = IREE_ASYNC_COMPLETION_FLAG_NONE;
  iree_async_operation_pool_t* pool = nullptr;
  iree_async_operation_t* acquired_during_callback = nullptr;
};

static void RecordCompletion(void* user_data, iree_async_operation_t* operation,
                             iree_status_t status,
                             iree_async_completion_flags_t flags) {
  (void)operation;
  CompletionRecord* record = static_cast<CompletionRecord*>(user_data);
  ++record->call_count;
  record->status_code = iree_status_code(status);
  record->flags = flags;
  iree_status_free(status);
  if (record->pool) {
    IREE_ASSERT_OK(iree_async_operation_pool_acquire(
        record->pool, sizeof(*operation), &record->acquired_during_callback));
  }
}

TEST(OperationCompletionTest, TransfersStatusToCallback) {
  CompletionRecord record;
  iree_async_operation_t operation;
  iree_async_operation_initialize(&operation, IREE_ASYNC_OPERATION_TYPE_NOP,
                                  IREE_ASYNC_OPERATION_FLAG_NONE,
                                  RecordCompletion, &record);

  EXPECT_EQ(iree_async_operation_complete(
                &operation,
                iree_make_status(IREE_STATUS_DATA_LOSS, "completion failure"),
                IREE_ASYNC_COMPLETION_FLAG_NONE),
            1u);
  EXPECT_EQ(record.call_count, 1u);
  EXPECT_EQ(record.status_code, IREE_STATUS_DATA_LOSS);
}

TEST(OperationCompletionTest, FreesSuppressedStatusWithoutCallback) {
  iree_async_operation_t operation;
  iree_async_operation_initialize(&operation, IREE_ASYNC_OPERATION_TYPE_MESSAGE,
                                  IREE_ASYNC_OPERATION_FLAG_NONE,
                                  /*completion_fn=*/nullptr,
                                  /*user_data=*/nullptr);

  EXPECT_EQ(iree_async_operation_complete(
                &operation,
                iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED, "pool full"),
                IREE_ASYNC_COMPLETION_FLAG_NONE),
            0u);
}

TEST(OperationCompletionTest, ResolvesCancellationAsSuccess) {
  CompletionRecord record;
  iree_async_operation_t operation;
  iree_async_operation_initialize(
      &operation, IREE_ASYNC_OPERATION_TYPE_NOP,
      IREE_ASYNC_OPERATION_FLAG_CANCELLATION_IS_SUCCESS, RecordCompletion,
      &record);

  EXPECT_EQ(iree_async_operation_complete(
                &operation, iree_status_from_code(IREE_STATUS_CANCELLED),
                IREE_ASYNC_COMPLETION_FLAG_NONE),
            1u);
  EXPECT_EQ(record.status_code, IREE_STATUS_OK);
  EXPECT_TRUE(
      iree_any_bit_set(record.flags, IREE_ASYNC_COMPLETION_FLAG_CANCELLED));
}

TEST(OperationCompletionTest, ReturnsFinalOperationToPoolAfterCallback) {
  iree_async_operation_pool_t* pool = nullptr;
  IREE_ASSERT_OK(iree_async_operation_pool_allocate(
      iree_async_operation_pool_options_default(), iree_allocator_system(),
      &pool));

  iree_async_operation_t* operation = nullptr;
  IREE_ASSERT_OK(
      iree_async_operation_pool_acquire(pool, sizeof(*operation), &operation));
  CompletionRecord record;
  record.pool = pool;
  iree_async_operation_initialize(operation, IREE_ASYNC_OPERATION_TYPE_NOP,
                                  IREE_ASYNC_OPERATION_FLAG_NONE,
                                  RecordCompletion, &record);
  operation->pool = pool;

  EXPECT_EQ(iree_async_operation_complete(operation, iree_ok_status(),
                                          IREE_ASYNC_COMPLETION_FLAG_NONE),
            1u);
  EXPECT_NE(record.acquired_during_callback, operation);

  iree_async_operation_t* recycled_operation = nullptr;
  IREE_ASSERT_OK(iree_async_operation_pool_acquire(
      pool, sizeof(*recycled_operation), &recycled_operation));
  EXPECT_EQ(recycled_operation, operation);
  iree_async_operation_pool_release(pool, recycled_operation);
  iree_async_operation_pool_release(pool, record.acquired_during_callback);
  iree_async_operation_pool_free(pool);
}

TEST(OperationCompletionTest, RetainsPooledMultishotUntilFinalCompletion) {
  iree_async_operation_pool_t* pool = nullptr;
  IREE_ASSERT_OK(iree_async_operation_pool_allocate(
      iree_async_operation_pool_options_default(), iree_allocator_system(),
      &pool));

  iree_async_operation_t* operation = nullptr;
  IREE_ASSERT_OK(
      iree_async_operation_pool_acquire(pool, sizeof(*operation), &operation));
  CompletionRecord record;
  iree_async_operation_initialize(operation, IREE_ASYNC_OPERATION_TYPE_NOP,
                                  IREE_ASYNC_OPERATION_FLAG_NONE,
                                  RecordCompletion, &record);
  operation->pool = pool;

  EXPECT_EQ(iree_async_operation_complete(operation, iree_ok_status(),
                                          IREE_ASYNC_COMPLETION_FLAG_MORE),
            1u);

  iree_async_operation_t* acquired_operation = nullptr;
  IREE_ASSERT_OK(iree_async_operation_pool_acquire(
      pool, sizeof(*acquired_operation), &acquired_operation));
  EXPECT_NE(acquired_operation, operation);
  iree_async_operation_pool_release(pool, acquired_operation);

  EXPECT_EQ(iree_async_operation_complete(operation, iree_ok_status(),
                                          IREE_ASYNC_COMPLETION_FLAG_NONE),
            1u);
  IREE_ASSERT_OK(iree_async_operation_pool_acquire(
      pool, sizeof(*acquired_operation), &acquired_operation));
  EXPECT_EQ(acquired_operation, operation);
  iree_async_operation_pool_release(pool, acquired_operation);
  iree_async_operation_pool_free(pool);
}

}  // namespace
