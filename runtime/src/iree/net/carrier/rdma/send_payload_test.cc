// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/carrier/rdma/send_payload.h"

#include <cstdint>
#include <cstring>
#include <vector>

#include "iree/net/carrier/rdma/sge.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

static void DestroyTestRegion(iree_async_region_t* region) {
  iree_allocator_free(iree_allocator_system(), region);
}

static iree_async_region_t* CreateTestRegion(iree_async_region_type_t type,
                                             void* base_ptr,
                                             iree_host_size_t length,
                                             iree_host_size_t buffer_size,
                                             iree_host_size_t buffer_count) {
  iree_async_region_t* region = nullptr;
  IREE_CHECK_OK(iree_allocator_malloc(iree_allocator_system(), sizeof(*region),
                                      (void**)&region));
  std::memset(region, 0, sizeof(*region));
  iree_atomic_ref_count_init(&region->ref_count);
  region->destroy_fn = DestroyTestRegion;
  region->type = type;
  region->base_ptr = base_ptr;
  region->length = length;
  region->buffer_size = buffer_size;
  region->buffer_count = buffer_count;
  region->handles.rdma.lkey = 0x12345678u;
  return region;
}

class RdmaSendPayloadTest : public ::testing::Test {
 protected:
  static constexpr iree_host_size_t kStagingBufferSize = 64;
  static constexpr iree_host_size_t kStagingBufferCount = 2;

  void SetUp() override {
    staging_storage_.resize(kStagingBufferSize * kStagingBufferCount);
    staging_region_ = CreateTestRegion(
        IREE_ASYNC_REGION_TYPE_RDMA, staging_storage_.data(),
        staging_storage_.size(), kStagingBufferSize, kStagingBufferCount);
    IREE_ASSERT_OK(iree_async_buffer_pool_create(
        staging_region_, iree_allocator_system(), &staging_pool_));
  }

  void TearDown() override {
    iree_async_buffer_pool_release(staging_pool_);
    iree_async_region_release(staging_region_);
  }

  std::vector<uint8_t> staging_storage_;
  iree_async_region_t* staging_region_ = nullptr;
  iree_async_buffer_pool_t* staging_pool_ = nullptr;
};

TEST_F(RdmaSendPayloadTest, StagesHeaderAndPreservesRegisteredPayload) {
  uint8_t header[] = {1, 2, 3, 4};
  std::vector<uint8_t> registered_storage(1024 * 1024, 0xCD);
  iree_async_region_t* registered_region =
      CreateTestRegion(IREE_ASYNC_REGION_TYPE_RDMA, registered_storage.data(),
                       registered_storage.size(), registered_storage.size(), 1);
  registered_region->handles.rdma.lkey = 0x87654321u;
  iree_async_span_t source_spans[] = {
      iree_async_span_from_ptr(header, sizeof(header)),
      iree_async_span_make(registered_region, 16,
                           registered_storage.size() - 16),
  };

  iree_net_rdma_send_payload_t payload;
  IREE_ASSERT_OK(iree_net_rdma_send_payload_prepare(
      iree_async_span_list_make(source_spans, IREE_ARRAYSIZE(source_spans)),
      IREE_NET_SEND_FLAG_NONE, staging_pool_, &payload));

  ASSERT_EQ(2u, payload.span_count);
  EXPECT_EQ(sizeof(header) + registered_storage.size() - 16,
            payload.byte_length);
  EXPECT_EQ(staging_region_, payload.spans[0].region);
  EXPECT_EQ(sizeof(header), payload.spans[0].length);
  EXPECT_EQ(registered_region, payload.spans[1].region);
  EXPECT_EQ(source_spans[1].offset, payload.spans[1].offset);
  EXPECT_EQ(source_spans[1].length, payload.spans[1].length);
  EXPECT_EQ(iree_async_span_ptr(source_spans[1]),
            iree_async_span_ptr(payload.spans[1]));

  struct ibv_sge scatter_gather_entries[2];
  int scatter_gather_entry_count = 0;
  IREE_ASSERT_OK(iree_net_rdma_sge_list_from_span_list(
      iree_async_span_list_make(payload.spans, payload.span_count),
      IREE_ARRAYSIZE(scatter_gather_entries), scatter_gather_entries,
      &scatter_gather_entry_count));
  ASSERT_EQ(2, scatter_gather_entry_count);
  EXPECT_EQ((uint64_t)(uintptr_t)iree_async_span_ptr(payload.spans[0]),
            scatter_gather_entries[0].addr);
  EXPECT_EQ(0x12345678u, scatter_gather_entries[0].lkey);
  EXPECT_EQ((uint64_t)(uintptr_t)iree_async_span_ptr(source_spans[1]),
            scatter_gather_entries[1].addr);
  EXPECT_EQ(0x87654321u, scatter_gather_entries[1].lkey);

  header[0] = 0xFF;
  const uint8_t* staged_header =
      static_cast<const uint8_t*>(iree_async_span_ptr(payload.spans[0]));
  EXPECT_EQ(1, staged_header[0]);

  iree_net_rdma_send_payload_deinitialize(&payload);
  iree_async_region_release(registered_region);
}

TEST_F(RdmaSendPayloadTest, PreservesInterleavedWireOrder) {
  uint8_t first_header[] = {1, 2};
  uint8_t second_header[] = {3, 4, 5};
  uint8_t registered_storage[] = {6, 7, 8, 9};
  iree_async_region_t* registered_region = CreateTestRegion(
      IREE_ASYNC_REGION_TYPE_RDMA, registered_storage,
      sizeof(registered_storage), sizeof(registered_storage), 1);
  iree_async_span_t source_spans[] = {
      iree_async_span_from_ptr(first_header, sizeof(first_header)),
      iree_async_span_make(registered_region, 0, sizeof(registered_storage)),
      iree_async_span_from_ptr(second_header, sizeof(second_header)),
  };

  iree_net_rdma_send_payload_t payload;
  IREE_ASSERT_OK(iree_net_rdma_send_payload_prepare(
      iree_async_span_list_make(source_spans, IREE_ARRAYSIZE(source_spans)),
      IREE_NET_SEND_FLAG_NONE, staging_pool_, &payload));

  ASSERT_EQ(3u, payload.span_count);
  EXPECT_EQ(staging_region_, payload.spans[0].region);
  EXPECT_EQ(registered_region, payload.spans[1].region);
  EXPECT_EQ(staging_region_, payload.spans[2].region);
  EXPECT_EQ(payload.spans[0].offset + payload.spans[0].length,
            payload.spans[2].offset);
  EXPECT_EQ(0, std::memcmp(iree_async_span_ptr(payload.spans[0]), first_header,
                           sizeof(first_header)));
  EXPECT_EQ(0, std::memcmp(iree_async_span_ptr(payload.spans[2]), second_header,
                           sizeof(second_header)));

  iree_net_rdma_send_payload_deinitialize(&payload);
  iree_async_region_release(registered_region);
}

TEST_F(RdmaSendPayloadTest, CoalescesAdjacentUnregisteredSpans) {
  uint8_t first[] = {1, 2};
  uint8_t second[] = {3, 4, 5};
  iree_async_span_t source_spans[] = {
      iree_async_span_from_ptr(first, sizeof(first)),
      iree_async_span_from_ptr(second, sizeof(second)),
  };

  iree_net_rdma_send_payload_t payload;
  IREE_ASSERT_OK(iree_net_rdma_send_payload_prepare(
      iree_async_span_list_make(source_spans, IREE_ARRAYSIZE(source_spans)),
      IREE_NET_SEND_FLAG_NONE, staging_pool_, &payload));

  ASSERT_EQ(1u, payload.span_count);
  EXPECT_EQ(sizeof(first) + sizeof(second), payload.spans[0].length);
  const uint8_t expected[] = {1, 2, 3, 4, 5};
  EXPECT_EQ(0, std::memcmp(iree_async_span_ptr(payload.spans[0]), expected,
                           sizeof(expected)));

  iree_net_rdma_send_payload_deinitialize(&payload);
}

TEST_F(RdmaSendPayloadTest, RegisteredSpansNeedNoStagingLease) {
  uint8_t registered_storage[16] = {0};
  iree_async_region_t* registered_region = CreateTestRegion(
      IREE_ASYNC_REGION_TYPE_RDMA, registered_storage,
      sizeof(registered_storage), sizeof(registered_storage), 1);
  iree_async_span_t source_span = iree_async_span_make(registered_region, 4, 8);

  iree_net_rdma_send_payload_t payload;
  IREE_ASSERT_OK(iree_net_rdma_send_payload_prepare(
      iree_async_span_list_make(&source_span, 1), IREE_NET_SEND_FLAG_ZERO_COPY,
      /*staging_pool=*/nullptr, &payload));

  ASSERT_EQ(1u, payload.span_count);
  EXPECT_EQ(registered_region, payload.spans[0].region);
  EXPECT_EQ(nullptr, payload.staging_buffer_lease.release.fn);

  iree_net_rdma_send_payload_deinitialize(&payload);
  iree_async_region_release(registered_region);
}

TEST_F(RdmaSendPayloadTest, ZeroCopyRejectsUnregisteredBytes) {
  uint8_t header[] = {1, 2, 3, 4};
  iree_async_span_t source_span =
      iree_async_span_from_ptr(header, sizeof(header));
  iree_host_size_t available_before =
      iree_async_buffer_pool_available(staging_pool_);
  iree_net_rdma_send_payload_t payload;

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_net_rdma_send_payload_prepare(
          iree_async_span_list_make(&source_span, 1),
          IREE_NET_SEND_FLAG_ZERO_COPY, staging_pool_, &payload));

  EXPECT_EQ(available_before, iree_async_buffer_pool_available(staging_pool_));
  EXPECT_EQ(nullptr, payload.staging_buffer_lease.release.fn);
}

TEST_F(RdmaSendPayloadTest, StagingLimitCountsOnlyUnregisteredBytes) {
  uint8_t header[kStagingBufferSize] = {0};
  std::vector<uint8_t> registered_storage(1024 * 1024, 0);
  iree_async_region_t* registered_region =
      CreateTestRegion(IREE_ASYNC_REGION_TYPE_RDMA, registered_storage.data(),
                       registered_storage.size(), registered_storage.size(), 1);
  iree_async_span_t source_spans[] = {
      iree_async_span_from_ptr(header, sizeof(header)),
      iree_async_span_make(registered_region, 0, registered_storage.size()),
  };

  iree_net_rdma_send_payload_t payload;
  IREE_ASSERT_OK(iree_net_rdma_send_payload_prepare(
      iree_async_span_list_make(source_spans, IREE_ARRAYSIZE(source_spans)),
      IREE_NET_SEND_FLAG_NONE, staging_pool_, &payload));

  EXPECT_EQ(sizeof(header) + registered_storage.size(), payload.byte_length);
  EXPECT_EQ(2u, payload.span_count);

  iree_net_rdma_send_payload_deinitialize(&payload);
  iree_async_region_release(registered_region);
}

}  // namespace
