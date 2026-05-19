// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/carrier/rdma/cm_channel.h"

#include <fcntl.h>

#include "iree/async/proactor_platform.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

struct RdmaCmLibrary {
  iree_net_librdmacm_t value;
};

void DeinitializeRdmaCmLibrary(RdmaCmLibrary* library) {
  iree_net_librdmacm_deinitialize(&library->value);
}

class RdmaCmChannelTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_status_t status = iree_net_librdmacm_initialize(
        iree_string_view_list_empty(), iree_allocator_system(),
        &librdmacm_.value);
    if (iree_status_code(status) == IREE_STATUS_NOT_FOUND ||
        iree_status_code(status) == IREE_STATUS_UNAVAILABLE) {
      iree_status_ignore(status);
      GTEST_SKIP() << "librdmacm is not available on this machine";
    }
    IREE_ASSERT_OK(status);

    IREE_ASSERT_OK(iree_async_proactor_create_platform(
        iree_async_proactor_options_default(), iree_allocator_system(),
        &proactor_));
  }

  void TearDown() override {
    iree_net_rdma_cm_channel_release(channel_);
    if (proactor_) iree_async_proactor_release(proactor_);
    DeinitializeRdmaCmLibrary(&librdmacm_);
  }

  static void OnEvent(void* user_data, iree_status_t status,
                      const iree_net_rdma_cm_event_t* event) {
    RdmaCmChannelTest* self = (RdmaCmChannelTest*)user_data;
    if (!iree_status_is_ok(status)) {
      ++self->error_count_;
      iree_status_ignore(status);
      return;
    }
    ++self->event_count_;
    EXPECT_NE(nullptr, event);
  }

  iree_net_rdma_cm_channel_callback_t callback() {
    iree_net_rdma_cm_channel_callback_t callback = {OnEvent, this};
    return callback;
  }

  iree_status_t CreateChannel() {
    return iree_net_rdma_cm_channel_create(&librdmacm_.value, proactor_,
                                           callback(), iree_allocator_system(),
                                           &channel_);
  }

  RdmaCmLibrary librdmacm_ = {};
  iree_async_proactor_t* proactor_ = nullptr;
  iree_net_rdma_cm_channel_t* channel_ = nullptr;
  int event_count_ = 0;
  int error_count_ = 0;
};

TEST_F(RdmaCmChannelTest, CreateSetsNonblockingAndCloseOnExec) {
  iree_status_t status = CreateChannel();
  if (iree_status_code(status) == IREE_STATUS_UNAVAILABLE) {
    iree_status_ignore(status);
    GTEST_SKIP() << "rdma_cm event channels are not available on this machine";
  }
  IREE_ASSERT_OK(status);

  struct rdma_event_channel* event_channel =
      iree_net_rdma_cm_channel_native_event_channel(channel_);
  ASSERT_NE(nullptr, event_channel);
  ASSERT_GE(event_channel->fd, 0);

  int flags = fcntl(event_channel->fd, F_GETFL);
  ASSERT_NE(-1, flags);
  EXPECT_NE(0, flags & O_NONBLOCK);

  int descriptor_flags = fcntl(event_channel->fd, F_GETFD);
  ASSERT_NE(-1, descriptor_flags);
  EXPECT_NE(0, descriptor_flags & FD_CLOEXEC);
}

TEST_F(RdmaCmChannelTest, DrainEmptyChannelDoesNotReportEvents) {
  iree_status_t status = CreateChannel();
  if (iree_status_code(status) == IREE_STATUS_UNAVAILABLE) {
    iree_status_ignore(status);
    GTEST_SKIP() << "rdma_cm event channels are not available on this machine";
  }
  IREE_ASSERT_OK(status);

  IREE_EXPECT_OK(iree_net_rdma_cm_channel_drain(channel_));
  EXPECT_EQ(0, event_count_);
  EXPECT_EQ(0, error_count_);
}

TEST_F(RdmaCmChannelTest, RejectsInvalidArguments) {
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_rdma_cm_channel_create(nullptr, proactor_, callback(),
                                      iree_allocator_system(), &channel_));

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_rdma_cm_channel_create(&librdmacm_.value, nullptr, callback(),
                                      iree_allocator_system(), &channel_));

  iree_net_rdma_cm_channel_callback_t null_callback = {nullptr, nullptr};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_net_rdma_cm_channel_create(
                            &librdmacm_.value, proactor_, null_callback,
                            iree_allocator_system(), &channel_));

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_rdma_cm_channel_create(&librdmacm_.value, proactor_, callback(),
                                      iree_allocator_system(), nullptr));

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_net_rdma_cm_channel_drain(nullptr));
}

}  // namespace
