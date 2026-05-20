// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/carrier/rdma/context.h"

#include <cstddef>
#include <cstdint>
#include <memory>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

struct RdmaContextDeleter {
  void operator()(iree_net_rdma_context_t* context) const {
    iree_net_rdma_context_release(context);
  }
};

using RdmaContextPtr =
    std::unique_ptr<iree_net_rdma_context_t, RdmaContextDeleter>;

bool ShouldSkipUnavailable(iree_status_code_t status_code) {
  return status_code == IREE_STATUS_NOT_FOUND ||
         status_code == IREE_STATUS_UNAVAILABLE;
}

iree_status_t CreateDefaultContext(iree_net_rdma_context_t** out_context) {
  return iree_net_rdma_context_create(iree_net_rdma_context_options_default(),
                                      iree_allocator_system(), out_context);
}

TEST(RdmaContextTest, CreateSelectsUsableDeviceAndPort) {
  iree_net_rdma_context_t* raw_context = nullptr;
  iree_status_t status = CreateDefaultContext(&raw_context);
  if (!iree_status_is_ok(status) &&
      ShouldSkipUnavailable(iree_status_code(status))) {
    iree_status_ignore(status);
    GTEST_SKIP() << "RDMA context is not available on this machine";
  }
  IREE_ASSERT_OK(status);
  RdmaContextPtr context(raw_context);

  EXPECT_NE(nullptr, iree_net_rdma_context_libverbs(context.get()));
  EXPECT_NE(nullptr, iree_net_rdma_context_librdmacm(context.get()));
  EXPECT_NE(nullptr, iree_net_rdma_context_device(context.get()));
  EXPECT_NE(nullptr, iree_net_rdma_context_protection_domain(context.get()));
  EXPECT_FALSE(iree_string_view_is_empty(
      iree_net_rdma_context_device_name(context.get())));
  EXPECT_NE(0, iree_net_rdma_context_port_number(context.get()));
  EXPECT_EQ(IBV_PORT_ACTIVE,
            iree_net_rdma_context_port_attributes(context.get())->state);
}

TEST(RdmaContextTest, RequestedMissingDeviceFailsLoudly) {
  iree_net_rdma_context_t* available_context = nullptr;
  iree_status_t available_status = CreateDefaultContext(&available_context);
  if (!iree_status_is_ok(available_status) &&
      ShouldSkipUnavailable(iree_status_code(available_status))) {
    iree_status_ignore(available_status);
    GTEST_SKIP() << "RDMA context is not available on this machine";
  }
  IREE_ASSERT_OK(available_status);
  iree_net_rdma_context_release(available_context);

  iree_net_rdma_context_options_t options =
      iree_net_rdma_context_options_default();
  options.device_name =
      iree_make_cstring_view("iree_missing_rdma_device_name_for_negative_test");

  iree_net_rdma_context_t* context = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_NOT_FOUND,
      iree_net_rdma_context_create(options, iree_allocator_system(), &context));
  EXPECT_EQ(nullptr, context);
}

TEST(RdmaContextTest, RegistersHostMemory) {
  iree_net_rdma_context_t* raw_context = nullptr;
  iree_status_t create_status = CreateDefaultContext(&raw_context);
  if (!iree_status_is_ok(create_status) &&
      ShouldSkipUnavailable(iree_status_code(create_status))) {
    iree_status_ignore(create_status);
    GTEST_SKIP() << "RDMA context is not available on this machine";
  }
  IREE_ASSERT_OK(create_status);
  RdmaContextPtr context(raw_context);

  constexpr iree_host_size_t kBufferSize = 4096;
  void* buffer = nullptr;
  IREE_ASSERT_OK(
      iree_allocator_malloc(iree_allocator_system(), kBufferSize, &buffer));

  struct ibv_mr* memory_region = nullptr;
  iree_status_t status = iree_net_rdma_context_register_host_memory(
      context.get(), buffer, kBufferSize,
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE,
      &memory_region);
  if (iree_status_is_ok(status)) {
    EXPECT_NE(nullptr, memory_region);
    IREE_EXPECT_OK(
        iree_net_rdma_context_deregister_memory(context.get(), memory_region));
  }
  iree_allocator_free(iree_allocator_system(), buffer);
  IREE_EXPECT_OK(status);
}

}  // namespace
