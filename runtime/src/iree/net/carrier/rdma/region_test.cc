// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/carrier/rdma/region.h"

#include <memory>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

struct RdmaContextDeleter {
  void operator()(iree_net_rdma_context_t* context) const {
    iree_net_rdma_context_release(context);
  }
};

struct AsyncSlabDeleter {
  void operator()(iree_async_slab_t* slab) const {
    iree_async_slab_release(slab);
  }
};

struct AsyncRegionDeleter {
  void operator()(iree_async_region_t* region) const {
    iree_async_region_release(region);
  }
};

using RdmaContextPtr =
    std::unique_ptr<iree_net_rdma_context_t, RdmaContextDeleter>;
using AsyncSlabPtr = std::unique_ptr<iree_async_slab_t, AsyncSlabDeleter>;
using AsyncRegionPtr = std::unique_ptr<iree_async_region_t, AsyncRegionDeleter>;

bool IsUnavailableStatus(iree_status_code_t status_code) {
  return status_code == IREE_STATUS_NOT_FOUND ||
         status_code == IREE_STATUS_UNAVAILABLE;
}

bool ConsumeUnavailableStatus(iree_status_t& status) {
  if (!IsUnavailableStatus(iree_status_code(status))) return false;
  iree::Status consumed_status = iree::internal::ConsumeForTest(status);
  (void)consumed_status;
  return true;
}

iree_async_slab_options_t SmallSlabOptions() {
  iree_async_slab_options_t options = iree_async_slab_options_default();
  options.buffer_size = 256;
  options.buffer_count = 4;
  return options;
}

TEST(RdmaRegionTest, RejectsInvalidArguments) {
  iree_async_slab_t* raw_slab = nullptr;
  IREE_ASSERT_OK(iree_async_slab_create(SmallSlabOptions(),
                                        iree_allocator_system(), &raw_slab));
  AsyncSlabPtr slab(raw_slab);

  iree_async_region_t* region = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_rdma_region_register_slab(nullptr, slab.get(),
                                         IREE_ASYNC_BUFFER_ACCESS_FLAG_READ,
                                         iree_allocator_system(), &region));
  EXPECT_EQ(nullptr, region);

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_rdma_region_register_slab(nullptr, slab.get(),
                                         IREE_ASYNC_BUFFER_ACCESS_FLAG_NONE,
                                         iree_allocator_system(), &region));
  EXPECT_EQ(nullptr, region);
}

TEST(RdmaRegionTest, RegistersSlabMemory) {
  iree_net_rdma_context_t* raw_context = nullptr;
  iree_status_t context_status =
      iree_net_rdma_context_create(iree_net_rdma_context_options_default(),
                                   iree_allocator_system(), &raw_context);
  if (ConsumeUnavailableStatus(context_status)) {
    GTEST_SKIP() << "RDMA context is not available on this machine";
  }
  IREE_ASSERT_OK(context_status);
  RdmaContextPtr context(raw_context);

  iree_async_region_t* null_slab_region = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_rdma_region_register_slab(
          context.get(), nullptr, IREE_ASYNC_BUFFER_ACCESS_FLAG_READ,
          iree_allocator_system(), &null_slab_region));
  EXPECT_EQ(nullptr, null_slab_region);

  iree_async_slab_t* raw_slab = nullptr;
  IREE_ASSERT_OK(iree_async_slab_create(SmallSlabOptions(),
                                        iree_allocator_system(), &raw_slab));
  AsyncSlabPtr slab(raw_slab);

  iree_async_region_t* raw_region = nullptr;
  IREE_ASSERT_OK(iree_net_rdma_region_register_slab(
      context.get(), slab.get(),
      IREE_ASYNC_BUFFER_ACCESS_FLAG_READ | IREE_ASYNC_BUFFER_ACCESS_FLAG_WRITE |
          IREE_ASYNC_BUFFER_ACCESS_FLAG_REMOTE_READ |
          IREE_ASYNC_BUFFER_ACCESS_FLAG_REMOTE_WRITE,
      iree_allocator_system(), &raw_region));
  AsyncRegionPtr region(raw_region);

  EXPECT_EQ(IREE_ASYNC_REGION_TYPE_RDMA, region->type);
  EXPECT_EQ(slab->base_ptr, region->base_ptr);
  EXPECT_EQ(slab->total_size, region->length);
  EXPECT_EQ(slab->buffer_size, region->buffer_size);
  EXPECT_EQ(slab->buffer_count, region->buffer_count);
  EXPECT_NE(nullptr, region->handles.rdma.mr);
}

}  // namespace
