// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/operations/message.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

static void ConsumeCompletion(void* user_data,
                              iree_async_operation_t* operation,
                              iree_status_t status,
                              iree_async_completion_flags_t flags) {
  iree_status_free(status);
}

static iree_async_message_operation_t MakeMessage() {
  iree_async_message_operation_t message = {};
  iree_async_operation_initialize(
      &message.base, IREE_ASYNC_OPERATION_TYPE_MESSAGE,
      IREE_ASYNC_OPERATION_FLAG_NONE, ConsumeCompletion, nullptr);
  message.target = reinterpret_cast<iree_async_proactor_t*>(uintptr_t{1});
  return message;
}

TEST(MessageOperationTest, AcceptsSourceCompletion) {
  iree_async_message_operation_t message = MakeMessage();
  IREE_EXPECT_OK(iree_async_message_operation_validate(&message));
}

TEST(MessageOperationTest, RequiresTarget) {
  iree_async_message_operation_t message = MakeMessage();
  message.target = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_async_message_operation_validate(&message));
}

TEST(MessageOperationTest, SourceCompletionRequiresCallback) {
  iree_async_message_operation_t message = MakeMessage();
  message.base.completion_fn = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_async_message_operation_validate(&message));
}

TEST(MessageOperationTest, AcceptsSuppressedSourceCompletion) {
  iree_async_message_operation_t message = MakeMessage();
  message.message_flags = IREE_ASYNC_MESSAGE_FLAG_SKIP_SOURCE_COMPLETION;
  message.base.completion_fn = nullptr;
  IREE_EXPECT_OK(iree_async_message_operation_validate(&message));
}

TEST(MessageOperationTest, SuppressedCompletionMustBeChainTail) {
  iree_async_message_operation_t message = MakeMessage();
  message.message_flags = IREE_ASYNC_MESSAGE_FLAG_SKIP_SOURCE_COMPLETION;
  message.base.flags = IREE_ASYNC_OPERATION_FLAG_LINKED;
  message.base.completion_fn = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_async_message_operation_validate(&message));
}

TEST(MessageOperationTest, SuppressedCompletionRejectsCallback) {
  iree_async_message_operation_t message = MakeMessage();
  message.message_flags = IREE_ASYNC_MESSAGE_FLAG_SKIP_SOURCE_COMPLETION;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_async_message_operation_validate(&message));
}

TEST(MessageOperationTest, SuppressedCompletionRejectsPool) {
  iree_async_message_operation_t message = MakeMessage();
  message.message_flags = IREE_ASYNC_MESSAGE_FLAG_SKIP_SOURCE_COMPLETION;
  message.base.completion_fn = nullptr;
  message.base.pool =
      reinterpret_cast<iree_async_operation_pool_t*>(uintptr_t{1});
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_async_message_operation_validate(&message));
}

}  // namespace
