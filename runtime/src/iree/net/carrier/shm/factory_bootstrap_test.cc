// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/carrier/shm/factory_bootstrap.h"

#if defined(IREE_PLATFORM_WINDOWS)
#include <windows.h>
#else
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#endif  // IREE_PLATFORM_WINDOWS

#include "iree/async/proactor_platform.h"
#include "iree/net/carrier/shm/factory.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

struct FailNthAllocator {
  iree_host_size_t allocation_count = 0;
  iree_host_size_t fail_allocation = 0;

  iree_allocator_t allocator() { return {this, Control}; }

  static iree_status_t Control(void* self, iree_allocator_command_t command,
                               const void* params, void** inout_ptr) {
    auto* allocator = static_cast<FailNthAllocator*>(self);
    switch (command) {
      case IREE_ALLOCATOR_COMMAND_MALLOC:
      case IREE_ALLOCATOR_COMMAND_CALLOC:
        ++allocator->allocation_count;
        if (allocator->allocation_count == allocator->fail_allocation) {
          return iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED);
        }
        break;
      default:
        break;
    }
    iree_allocator_t system_allocator = iree_allocator_system();
    return system_allocator.ctl(system_allocator.self, command, params,
                                inout_ptr);
  }
};

static void RecordUnexpectedCompletion(
    void* user_data, iree_status_t status,
    iree_net_shm_bootstrap_completion_flags_t flags,
    iree_net_connection_t* connection) {
  *static_cast<bool*>(user_data) = true;
  iree_status_free(status);
  iree_net_connection_release(connection);
  (void)flags;
}

class FactoryBootstrapTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_status_t status = iree_async_proactor_create_platform(
        iree_async_proactor_options_default(), iree_allocator_system(),
        &proactor_);
    if (iree_status_is_unavailable(status)) {
      iree_status_free(status);
      GTEST_SKIP() << "Platform proactor unavailable";
    }
    IREE_ASSERT_OK(status);

    IREE_ASSERT_OK(
        iree_net_shm_factory_create(iree_net_shm_carrier_options_default(),
                                    iree_allocator_system(), &base_factory_));
    factory_ = reinterpret_cast<iree_net_shm_factory_t*>(base_factory_);
  }

  void TearDown() override {
    iree_net_transport_factory_release(base_factory_);
    iree_async_proactor_release(proactor_);
  }

  iree_async_proactor_t* proactor_ = nullptr;
  iree_net_transport_factory_t* base_factory_ = nullptr;
  iree_net_shm_factory_t* factory_ = nullptr;
};

TEST_F(FactoryBootstrapTest, PrepareFailureRetainsCallerChannel) {
#if defined(IREE_PLATFORM_WINDOWS)
  HANDLE handle = CreateEventW(/*lpEventAttributes=*/nullptr,
                               /*bManualReset=*/TRUE,
                               /*bInitialState=*/FALSE,
                               /*lpName=*/nullptr);
  ASSERT_NE(handle, nullptr);
  iree_async_primitive_t channel =
      iree_async_primitive_from_win32_handle((uintptr_t)handle);
#else
  int pipe_fds[2] = {-1, -1};
  ASSERT_EQ(pipe(pipe_fds), 0) << strerror(errno);
  const int channel_fd = pipe_fds[0];
  iree_async_primitive_t channel = iree_async_primitive_from_fd(channel_fd);
#endif  // IREE_PLATFORM_WINDOWS

  FailNthAllocator allocator;
  allocator.fail_allocation = 2;
  bool callback_called = false;
  iree_net_shm_bootstrap_t* bootstrap = nullptr;
  iree_net_shm_bootstrap_callback_t callback = {
      /*.fn=*/RecordUnexpectedCompletion,
      /*.user_data=*/&callback_called,
  };
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_net_shm_bootstrap_prepare(
          factory_, IREE_NET_SHM_BOOTSTRAP_ROLE_SERVER, &channel, proactor_,
          callback, allocator.allocator(), &bootstrap));

  EXPECT_EQ(allocator.allocation_count, 2u);
  EXPECT_EQ(bootstrap, nullptr);
  EXPECT_FALSE(callback_called);
  EXPECT_FALSE(iree_async_primitive_is_none(channel));
#if defined(IREE_PLATFORM_WINDOWS)
  DWORD handle_flags = 0;
  EXPECT_TRUE(GetHandleInformation(handle, &handle_flags));
#else
  errno = 0;
  EXPECT_NE(fcntl(channel_fd, F_GETFD), -1) << strerror(errno);
#endif  // IREE_PLATFORM_WINDOWS

  iree_async_primitive_close(&channel);
#if !defined(IREE_PLATFORM_WINDOWS)
  close(pipe_fds[1]);
#endif  // !IREE_PLATFORM_WINDOWS
}

}  // namespace
