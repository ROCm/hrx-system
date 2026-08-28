// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/util/continuation.h"

#include <vector>

#include "iree/async/operation.h"
#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

struct CompletionRecord {
  iree_async_operation_t* operation;
  iree_status_code_t status_code;
  iree_async_completion_flags_t flags;
};

struct TestContext {
  iree_status_code_t submit_status_code = IREE_STATUS_OK;
  iree_async_operation_t* submitted_chain = nullptr;
  std::vector<CompletionRecord> completions;
};

static void RecordCompletion(void* user_data, iree_async_operation_t* operation,
                             iree_status_t status,
                             iree_async_completion_flags_t flags) {
  TestContext* context = static_cast<TestContext*>(user_data);
  context->completions.push_back({operation, iree_status_code(status), flags});
  iree_status_free(status);
}

static iree_status_t SubmitChain(void* user_data,
                                 iree_async_operation_t* chain_head) {
  TestContext* context = static_cast<TestContext*>(user_data);
  context->submitted_chain = chain_head;
  return iree_status_from_code(context->submit_status_code);
}

class ContinuationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(operations_); ++i) {
      iree_async_operation_initialize(
          &operations_[i], IREE_ASYNC_OPERATION_TYPE_NOP,
          IREE_ASYNC_OPERATION_FLAG_NONE, RecordCompletion, &context_);
    }
  }

  void Link(iree_host_size_t from, iree_host_size_t to) {
    operations_[from].linked_next = &operations_[to];
  }

  TestContext context_;
  iree_async_operation_t operations_[3];
};

TEST_F(ContinuationTest, PrepareBatchBuildsIntrusiveChains) {
  operations_[0].flags = IREE_ASYNC_OPERATION_FLAG_LINKED;
  operations_[1].flags = IREE_ASYNC_OPERATION_FLAG_LINKED;
  iree_async_operation_t* operation_ptrs[] = {&operations_[0], &operations_[1],
                                              &operations_[2]};

  IREE_ASSERT_OK(
      iree_async_continuation_prepare_batch(iree_async_operation_list_make(
          operation_ptrs, IREE_ARRAYSIZE(operation_ptrs))));

  EXPECT_EQ(operations_[0].linked_next, &operations_[1]);
  EXPECT_EQ(operations_[1].linked_next, &operations_[2]);
  EXPECT_EQ(operations_[2].linked_next, nullptr);
}

TEST_F(ContinuationTest, PrepareBatchRejectsNullOperationWithoutMutation) {
  operations_[0].linked_next = &operations_[2];
  iree_async_operation_t* operation_ptrs[] = {&operations_[0], nullptr};

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_async_continuation_prepare_batch(iree_async_operation_list_make(
          operation_ptrs, IREE_ARRAYSIZE(operation_ptrs))));
  EXPECT_EQ(operations_[0].linked_next, &operations_[2]);
}

TEST_F(ContinuationTest, PrepareBatchRejectsLinkedFinalOperation) {
  operations_[1].flags = IREE_ASYNC_OPERATION_FLAG_LINKED;
  operations_[0].linked_next = &operations_[2];
  iree_async_operation_t* operation_ptrs[] = {&operations_[0], &operations_[1]};

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_async_continuation_prepare_batch(iree_async_operation_list_make(
          operation_ptrs, IREE_ARRAYSIZE(operation_ptrs))));
  EXPECT_EQ(operations_[0].linked_next, &operations_[2]);
}

TEST_F(ContinuationTest, TakeDetachesChain) {
  Link(0, 1);
  EXPECT_EQ(iree_async_continuation_take(&operations_[0]), &operations_[1]);
  EXPECT_EQ(operations_[0].linked_next, nullptr);
}

TEST_F(ContinuationTest, EmptyChainReturnsZero) {
  iree_async_continuation_t continuation = iree_async_continuation_begin(
      SubmitChain, &context_, nullptr, IREE_STATUS_OK);
  EXPECT_EQ(iree_async_continuation_finish(&continuation), 0u);
  EXPECT_EQ(context_.submitted_chain, nullptr);
  EXPECT_TRUE(context_.completions.empty());
}

TEST_F(ContinuationTest, SuccessfulTriggerSubmitsChain) {
  Link(0, 1);

  iree_async_continuation_t continuation = iree_async_continuation_begin(
      SubmitChain, &context_, &operations_[0], IREE_STATUS_OK);
  EXPECT_EQ(context_.submitted_chain, &operations_[0]);
  EXPECT_EQ(operations_[0].linked_next, &operations_[1]);
  EXPECT_TRUE(context_.completions.empty());
  EXPECT_EQ(iree_async_continuation_finish(&continuation), 0u);
}

TEST_F(ContinuationTest, FailedTriggerCancelsEntireChain) {
  Link(0, 1);
  Link(1, 2);

  iree_async_continuation_t continuation = iree_async_continuation_begin(
      SubmitChain, &context_, &operations_[0], IREE_STATUS_INTERNAL);
  EXPECT_EQ(context_.submitted_chain, nullptr);
  EXPECT_TRUE(context_.completions.empty());
  EXPECT_EQ(iree_async_continuation_finish(&continuation), 3u);
  ASSERT_EQ(context_.completions.size(), 3u);
  for (iree_host_size_t i = 0; i < context_.completions.size(); ++i) {
    EXPECT_EQ(context_.completions[i].operation, &operations_[i]);
    EXPECT_EQ(context_.completions[i].status_code, IREE_STATUS_CANCELLED);
    EXPECT_EQ(context_.completions[i].flags, IREE_ASYNC_COMPLETION_FLAG_NONE);
    EXPECT_EQ(operations_[i].linked_next, nullptr);
  }
}

TEST_F(ContinuationTest, SubmitFailureCompletesHeadAndCancelsTail) {
  Link(0, 1);
  Link(1, 2);
  context_.submit_status_code = IREE_STATUS_RESOURCE_EXHAUSTED;

  iree_async_continuation_t continuation = iree_async_continuation_begin(
      SubmitChain, &context_, &operations_[0], IREE_STATUS_OK);
  EXPECT_EQ(context_.submitted_chain, &operations_[0]);
  EXPECT_TRUE(context_.completions.empty());
  EXPECT_EQ(iree_async_continuation_finish(&continuation), 3u);
  ASSERT_EQ(context_.completions.size(), 3u);
  EXPECT_EQ(context_.completions[0].operation, &operations_[0]);
  EXPECT_EQ(context_.completions[0].status_code,
            IREE_STATUS_RESOURCE_EXHAUSTED);
  EXPECT_EQ(context_.completions[1].operation, &operations_[1]);
  EXPECT_EQ(context_.completions[1].status_code, IREE_STATUS_CANCELLED);
  EXPECT_EQ(context_.completions[2].operation, &operations_[2]);
  EXPECT_EQ(context_.completions[2].status_code, IREE_STATUS_CANCELLED);
}

TEST_F(ContinuationTest, SuppressedCompletionIsNotCounted) {
  Link(0, 1);
  Link(1, 2);
  operations_[1].completion_fn = nullptr;

  iree_async_continuation_t continuation = iree_async_continuation_begin(
      SubmitChain, &context_, &operations_[0], IREE_STATUS_ABORTED);
  EXPECT_TRUE(context_.completions.empty());
  EXPECT_EQ(iree_async_continuation_finish(&continuation), 2u);
  ASSERT_EQ(context_.completions.size(), 2u);
  EXPECT_EQ(context_.completions[0].operation, &operations_[0]);
  EXPECT_EQ(context_.completions[1].operation, &operations_[2]);
}

TEST_F(ContinuationTest, FailedSuppressedTailIsConsumedDuringBegin) {
  operations_[0].completion_fn = nullptr;
  context_.submit_status_code = IREE_STATUS_RESOURCE_EXHAUSTED;

  iree_async_continuation_t continuation = iree_async_continuation_begin(
      SubmitChain, &context_, &operations_[0], IREE_STATUS_OK);

  EXPECT_EQ(context_.submitted_chain, &operations_[0]);
  EXPECT_EQ(continuation.chain_head, nullptr);
  EXPECT_EQ(iree_async_continuation_finish(&continuation), 0u);
  EXPECT_TRUE(context_.completions.empty());
}

TEST_F(ContinuationTest, CancelledSuppressedTailIsConsumedDuringBegin) {
  operations_[0].completion_fn = nullptr;

  iree_async_continuation_t continuation = iree_async_continuation_begin(
      SubmitChain, &context_, &operations_[0], IREE_STATUS_ABORTED);

  EXPECT_EQ(context_.submitted_chain, nullptr);
  EXPECT_EQ(continuation.chain_head, nullptr);
  EXPECT_EQ(iree_async_continuation_finish(&continuation), 0u);
  EXPECT_TRUE(context_.completions.empty());
}

}  // namespace
