// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/carrier/rdma/credit_memory.h"

#include <cstdint>
#include <memory>

#include "iree/base/internal/atomics.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

struct RdmaContextDeleter {
  void operator()(iree_net_rdma_context_t* context) const {
    iree_net_rdma_context_release(context);
  }
};

struct RdmaCreditMemoryDeleter {
  void operator()(iree_net_rdma_credit_memory_t* memory) const {
    iree_net_rdma_credit_memory_release(memory);
  }
};

using RdmaContextPtr =
    std::unique_ptr<iree_net_rdma_context_t, RdmaContextDeleter>;
using RdmaCreditMemoryPtr =
    std::unique_ptr<iree_net_rdma_credit_memory_t, RdmaCreditMemoryDeleter>;

bool ConsumeUnavailableStatus(iree_status_t& status) {
  iree_status_code_t code = iree_status_code(status);
  if (code != IREE_STATUS_NOT_FOUND && code != IREE_STATUS_UNAVAILABLE) {
    return false;
  }
  iree::Status consumed_status = iree::internal::ConsumeForTest(status);
  (void)consumed_status;
  return true;
}

TEST(RdmaCreditMemoryTest, RejectsNullContext) {
  iree_net_rdma_credit_memory_t* memory = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_rdma_credit_memory_create(nullptr, /*initial_credit_limit=*/0,
                                         iree_allocator_system(), &memory));
  EXPECT_EQ(nullptr, memory);
}

TEST(RdmaCreditMemoryTest, StoreSgeRejectsMissingStorage) {
  struct ibv_sge scatter_gather_entry;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_rdma_credit_memory_store_sge(nullptr, 1, &scatter_gather_entry));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_rdma_credit_memory_store_sge(nullptr, 1, nullptr));
}

TEST(RdmaCreditMemoryTest, PublishesRegisteredCounter) {
  iree_net_rdma_context_t* raw_context = nullptr;
  iree_status_t context_status =
      iree_net_rdma_context_create(iree_net_rdma_context_options_default(),
                                   iree_allocator_system(), &raw_context);
  if (ConsumeUnavailableStatus(context_status)) {
    GTEST_SKIP() << "RDMA context is not available on this machine";
  }
  IREE_ASSERT_OK(context_status);
  RdmaContextPtr context(raw_context);

  iree_net_rdma_credit_memory_t* raw_memory = nullptr;
  IREE_ASSERT_OK(iree_net_rdma_credit_memory_create(
      context.get(), /*initial_credit_limit=*/7, iree_allocator_system(),
      &raw_memory));
  RdmaCreditMemoryPtr memory(raw_memory);

  EXPECT_EQ(7u, iree_net_rdma_credit_memory_load(memory.get()));
  iree_net_rdma_credit_memory_store(memory.get(), 11);
  EXPECT_EQ(11u, iree_net_rdma_credit_memory_load(memory.get()));

  iree_net_rdma_remote_credit_memory_t remote_memory =
      iree_net_rdma_credit_memory_remote(memory.get());
  EXPECT_NE(0u, remote_memory.address);
  EXPECT_EQ(sizeof(uint32_t), remote_memory.length);

  auto* counter =
      reinterpret_cast<iree_atomic_uint32_t*>((uintptr_t)remote_memory.address);
  iree_atomic_store(counter, 13u, iree_memory_order_release);
  EXPECT_EQ(13u, iree_net_rdma_credit_memory_load(memory.get()));

  struct ibv_sge scatter_gather_entry;
  IREE_ASSERT_OK(iree_net_rdma_credit_memory_store_sge(memory.get(), 17,
                                                       &scatter_gather_entry));
  EXPECT_EQ(17u, iree_net_rdma_credit_memory_load(memory.get()));
  EXPECT_EQ(remote_memory.address, scatter_gather_entry.addr);
  EXPECT_EQ(sizeof(uint32_t), scatter_gather_entry.length);
  EXPECT_NE(0u, scatter_gather_entry.lkey);
}

TEST(RdmaCreditMemoryTest, EmptyRemoteDescriptorForNullMemory) {
  iree_net_rdma_remote_credit_memory_t remote_memory =
      iree_net_rdma_credit_memory_remote(nullptr);
  EXPECT_EQ(0u, remote_memory.address);
  EXPECT_EQ(0u, remote_memory.rkey);
  EXPECT_EQ(0u, remote_memory.length);
}

}  // namespace
