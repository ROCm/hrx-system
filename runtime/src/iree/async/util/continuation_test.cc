// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/util/continuation.h"

#include <vector>

#include "iree/async/operation.h"
#include "iree/async/operations/file.h"
#include "iree/async/region.h"
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

struct ReuseContext {
  iree_host_size_t call_count = 0;
};

static void ReinitializeOnCompletion(void* user_data,
                                     iree_async_operation_t* operation,
                                     iree_status_t status,
                                     iree_async_completion_flags_t flags) {
  (void)flags;
  ReuseContext* context = static_cast<ReuseContext*>(user_data);
  ++context->call_count;
  iree_status_free(status);
  iree_async_operation_initialize(operation, IREE_ASYNC_OPERATION_TYPE_NOP,
                                  IREE_ASYNC_OPERATION_FLAG_NONE,
                                  ReinitializeOnCompletion, context);
}

struct TestRegion {
  iree_async_region_t base;
  bool destroyed = false;
};

static void DestroyTestRegion(iree_async_region_t* base_region) {
  auto* region = reinterpret_cast<TestRegion*>(base_region);
  region->destroyed = true;
}

struct RegionCancellationContext {
  TestRegion* region = nullptr;
  bool callback_fired = false;
};

static void RecordRegionCancellation(void* user_data,
                                     iree_async_operation_t* operation,
                                     iree_status_t status,
                                     iree_async_completion_flags_t flags) {
  (void)operation;
  (void)flags;
  auto* context = static_cast<RegionCancellationContext*>(user_data);
  context->callback_fired = true;
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_CANCELLED);
  EXPECT_FALSE(context->region->destroyed);
  EXPECT_EQ(iree_atomic_ref_count_load(&context->region->base.ref_count), 1);
  iree_status_free(status);
}

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
  iree_async_operation_t operations_[5];
};

TEST_F(ContinuationTest, ChainIteratorAdvancesBeforeReturningHead) {
  ReuseContext reuse_context;
  operations_[0].flags = IREE_ASYNC_OPERATION_FLAG_LINKED;
  operations_[0].completion_fn = ReinitializeOnCompletion;
  operations_[0].user_data = &reuse_context;
  operations_[2].flags = IREE_ASYNC_OPERATION_FLAG_LINKED;
  operations_[2].completion_fn = ReinitializeOnCompletion;
  operations_[2].user_data = &reuse_context;
  operations_[4].completion_fn = ReinitializeOnCompletion;
  operations_[4].user_data = &reuse_context;
  iree_async_operation_t* operation_ptrs[] = {&operations_[0], &operations_[1],
                                              &operations_[2], &operations_[3],
                                              &operations_[4]};
  iree_async_operation_list_t operation_list = iree_async_operation_list_make(
      operation_ptrs, IREE_ARRAYSIZE(operation_ptrs));
  IREE_ASSERT_OK(iree_async_continuation_prepare_batch(operation_list));

  iree_async_continuation_chain_iterator_t iterator =
      iree_async_continuation_chain_iterator_make(operation_list);
  iree_async_operation_t* first_head =
      iree_async_continuation_chain_iterator_next(&iterator);
  EXPECT_EQ(first_head, &operations_[0]);

  // Model a final callback immediately reusing the published chain head. The
  // iterator must have already consumed its original LINKED flag.
  first_head->completion_fn(first_head->user_data, first_head, iree_ok_status(),
                            IREE_ASYNC_COMPLETION_FLAG_NONE);
  iree_async_operation_t* second_head =
      iree_async_continuation_chain_iterator_next(&iterator);
  EXPECT_EQ(second_head, &operations_[2]);

  second_head->completion_fn(second_head->user_data, second_head,
                             iree_ok_status(), IREE_ASYNC_COMPLETION_FLAG_NONE);
  iree_async_operation_t* third_head =
      iree_async_continuation_chain_iterator_next(&iterator);
  EXPECT_EQ(third_head, &operations_[4]);
  third_head->completion_fn(third_head->user_data, third_head, iree_ok_status(),
                            IREE_ASYNC_COMPLETION_FLAG_NONE);
  EXPECT_EQ(iree_async_continuation_chain_iterator_next(&iterator), nullptr);
  EXPECT_EQ(reuse_context.call_count, 3u);
}

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

TEST_F(ContinuationTest, CancellationReleasesAcceptedRegionAfterCallback) {
  TestRegion region = {};
  iree_atomic_ref_count_init(&region.base.ref_count);
  region.base.destroy_fn = DestroyTestRegion;

  RegionCancellationContext context = {};
  context.region = &region;
  iree_async_file_read_operation_t read_op = {};
  iree_async_operation_initialize(
      &read_op.base, IREE_ASYNC_OPERATION_TYPE_FILE_READ,
      IREE_ASYNC_OPERATION_FLAG_NONE, RecordRegionCancellation, &context);
  read_op.buffer = iree_async_span_make(&region.base, 0, 1);
  iree_async_operation_acquire_resources(&read_op.base);
  iree_async_region_release(&region.base);

  EXPECT_EQ(iree_async_continuation_cancel(&read_op.base), 1u);
  EXPECT_TRUE(context.callback_fired);
  EXPECT_TRUE(region.destroyed);
  EXPECT_FALSE(read_op.base.resources_acquired);
  EXPECT_EQ(read_op.base.acquired_span_count, 0);
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
