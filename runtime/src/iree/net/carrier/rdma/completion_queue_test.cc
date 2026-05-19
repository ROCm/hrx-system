// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/carrier/rdma/completion_queue.h"

#include <fcntl.h>

#include <memory>

#include "iree/async/proactor_platform.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

struct RdmaContextDeleter {
  void operator()(iree_net_rdma_context_t* context) const {
    iree_net_rdma_context_release(context);
  }
};

struct RdmaCompletionQueueDeleter {
  void operator()(iree_net_rdma_completion_queue_t* queue) const {
    iree_net_rdma_completion_queue_release(queue);
  }
};

struct ProactorDeleter {
  void operator()(iree_async_proactor_t* proactor) const {
    iree_async_proactor_release(proactor);
  }
};

using RdmaCompletionQueuePtr = std::unique_ptr<iree_net_rdma_completion_queue_t,
                                               RdmaCompletionQueueDeleter>;
using RdmaContextPtr =
    std::unique_ptr<iree_net_rdma_context_t, RdmaContextDeleter>;
using ProactorPtr = std::unique_ptr<iree_async_proactor_t, ProactorDeleter>;

bool IsUnavailableStatus(iree_status_t status) {
  iree_status_code_t code = iree_status_code(status);
  return code == IREE_STATUS_NOT_FOUND || code == IREE_STATUS_UNAVAILABLE;
}

bool ConsumeIfUnavailable(iree_status_t status) {
  if (IsUnavailableStatus(status)) {
    iree_status_consume_code(status);
    return true;
  }
  return false;
}

iree_net_rdma_completion_queue_callback_t NoopCallback() {
  iree_net_rdma_completion_queue_callback_t callback = {
      [](void* user_data, iree_status_t status,
         const struct ibv_wc* completions, iree_host_size_t completion_count) {
        (void)user_data;
        (void)completions;
        (void)completion_count;
        if (!iree_status_is_ok(status)) {
          IREE_EXPECT_OK(status);
        }
      },
      nullptr,
  };
  return callback;
}

TEST(RdmaCompletionQueueTest, CreateSetsNativeResourcesPollable) {
  iree_net_rdma_context_t* raw_context = nullptr;
  iree_status_t context_status =
      iree_net_rdma_context_create(iree_net_rdma_context_options_default(),
                                   iree_allocator_system(), &raw_context);
  if (ConsumeIfUnavailable(context_status)) {
    GTEST_SKIP() << "RDMA context is not available";
  }
  IREE_ASSERT_OK(context_status);
  RdmaContextPtr context(raw_context);

  iree_async_proactor_t* raw_proactor = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_create_platform(
      iree_async_proactor_options_default(), iree_allocator_system(),
      &raw_proactor));
  ProactorPtr proactor(raw_proactor);

  iree_net_rdma_completion_queue_options_t options =
      iree_net_rdma_completion_queue_options_default();
  options.completion_capacity = 8;

  iree_net_rdma_completion_queue_t* raw_queue = nullptr;
  IREE_ASSERT_OK(iree_net_rdma_completion_queue_create(
      context.get(), proactor.get(), options, NoopCallback(),
      iree_allocator_system(), &raw_queue));
  RdmaCompletionQueuePtr queue(raw_queue);

  struct ibv_comp_channel* native_channel =
      iree_net_rdma_completion_queue_native_channel(queue.get());
  EXPECT_NE(nullptr, native_channel);
  EXPECT_NE(nullptr, iree_net_rdma_completion_queue_native_cq(queue.get()));
  ASSERT_GE(native_channel->fd, 0);

  int flags = fcntl(native_channel->fd, F_GETFL);
  ASSERT_NE(-1, flags);
  EXPECT_NE(0, flags & O_NONBLOCK);

  int descriptor_flags = fcntl(native_channel->fd, F_GETFD);
  ASSERT_NE(-1, descriptor_flags);
  EXPECT_NE(0, descriptor_flags & FD_CLOEXEC);

  IREE_EXPECT_OK(iree_net_rdma_completion_queue_drain(queue.get()));
}

TEST(RdmaCompletionQueueTest, RejectsInvalidArguments) {
  iree_async_proactor_t* raw_proactor = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_create_platform(
      iree_async_proactor_options_default(), iree_allocator_system(),
      &raw_proactor));
  ProactorPtr proactor(raw_proactor);

  iree_net_rdma_completion_queue_options_t options =
      iree_net_rdma_completion_queue_options_default();
  iree_net_rdma_completion_queue_t* queue = nullptr;

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_net_rdma_completion_queue_create(
                            nullptr, proactor.get(), options, NoopCallback(),
                            iree_allocator_system(), &queue));

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_net_rdma_completion_queue_drain(nullptr));

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_net_rdma_completion_queue_activate(nullptr));

  EXPECT_EQ(nullptr, iree_net_rdma_completion_queue_native_channel(nullptr));
  EXPECT_EQ(nullptr, iree_net_rdma_completion_queue_native_cq(nullptr));
}

TEST(RdmaCompletionQueueTest, RejectsInvalidOptionsAndCallback) {
  iree_net_rdma_context_t* raw_context = nullptr;
  iree_status_t context_status =
      iree_net_rdma_context_create(iree_net_rdma_context_options_default(),
                                   iree_allocator_system(), &raw_context);
  if (ConsumeIfUnavailable(context_status)) {
    GTEST_SKIP() << "RDMA context is not available";
  }
  IREE_ASSERT_OK(context_status);
  RdmaContextPtr context(raw_context);

  iree_async_proactor_t* raw_proactor = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_create_platform(
      iree_async_proactor_options_default(), iree_allocator_system(),
      &raw_proactor));
  ProactorPtr proactor(raw_proactor);

  iree_net_rdma_completion_queue_options_t options =
      iree_net_rdma_completion_queue_options_default();
  iree_net_rdma_completion_queue_t* queue = nullptr;

  iree_net_rdma_completion_queue_options_t invalid_capacity = options;
  invalid_capacity.completion_capacity = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_net_rdma_completion_queue_create(
                            context.get(), proactor.get(), invalid_capacity,
                            NoopCallback(), iree_allocator_system(), &queue));

  iree_net_rdma_completion_queue_options_t invalid_vector = options;
  invalid_vector.completion_vector = -1;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_net_rdma_completion_queue_create(
                            context.get(), proactor.get(), invalid_vector,
                            NoopCallback(), iree_allocator_system(), &queue));

  iree_net_rdma_completion_queue_options_t invalid_flags = options;
  invalid_flags.flags = 0x80u;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_net_rdma_completion_queue_create(
                            context.get(), proactor.get(), invalid_flags,
                            NoopCallback(), iree_allocator_system(), &queue));

  iree_net_rdma_completion_queue_callback_t null_callback = {nullptr, nullptr};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_net_rdma_completion_queue_create(
                            context.get(), proactor.get(), options,
                            null_callback, iree_allocator_system(), &queue));

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_net_rdma_completion_queue_create(
                            context.get(), nullptr, options, NoopCallback(),
                            iree_allocator_system(), &queue));
}

}  // namespace
