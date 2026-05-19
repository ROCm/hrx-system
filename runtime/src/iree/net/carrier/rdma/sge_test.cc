// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/carrier/rdma/sge.h"

#include <cstdint>
#include <cstring>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

iree_async_region_t MakeRegion(void* base_ptr, iree_host_size_t length) {
  iree_async_region_t region;
  std::memset(&region, 0, sizeof(region));
  region.type = IREE_ASYNC_REGION_TYPE_RDMA;
  region.base_ptr = base_ptr;
  region.length = length;
  region.handles.rdma.lkey = 0x12345678u;
  region.handles.rdma.rkey = 0x87654321u;
  return region;
}

TEST(RdmaSgeTest, ConvertsSpan) {
  char storage[64] = {0};
  iree_async_region_t region = MakeRegion(storage, sizeof(storage));
  iree_async_span_t span = {&region, /*offset=*/4, /*length=*/16};

  struct ibv_sge sge;
  IREE_ASSERT_OK(iree_net_rdma_sge_from_span(span, &sge));

  EXPECT_EQ((uint64_t)(uintptr_t)(storage + 4), sge.addr);
  EXPECT_EQ(16u, sge.length);
  EXPECT_EQ(0x12345678u, sge.lkey);
}

TEST(RdmaSgeTest, RejectsInvalidSpan) {
  char storage[64] = {0};
  iree_async_region_t region = MakeRegion(storage, sizeof(storage));
  struct ibv_sge sge;

  iree_async_span_t null_region_span = {nullptr, /*offset=*/0, /*length=*/1};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_net_rdma_sge_from_span(null_region_span, &sge));

  iree_async_region_t non_rdma_region = region;
  non_rdma_region.type = IREE_ASYNC_REGION_TYPE_NONE;
  iree_async_span_t non_rdma_span = {&non_rdma_region, /*offset=*/0,
                                     /*length=*/1};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_net_rdma_sge_from_span(non_rdma_span, &sge));

  iree_async_region_t device_region = region;
  device_region.base_ptr = nullptr;
  iree_async_span_t device_span = {&device_region, /*offset=*/0, /*length=*/1};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED,
                        iree_net_rdma_sge_from_span(device_span, &sge));

  iree_async_span_t out_of_range_span = {&region, /*offset=*/63,
                                         /*length=*/2};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        iree_net_rdma_sge_from_span(out_of_range_span, &sge));

  iree_async_region_t large_region =
      MakeRegion(storage, (iree_host_size_t)UINT32_MAX + 1);
  iree_async_span_t too_large_span = {&large_region, /*offset=*/0,
                                      (iree_host_size_t)UINT32_MAX + 1};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        iree_net_rdma_sge_from_span(too_large_span, &sge));
}

TEST(RdmaSgeTest, ConvertsSpanList) {
  char storage[64] = {0};
  iree_async_region_t region = MakeRegion(storage, sizeof(storage));
  iree_async_span_t spans[2] = {
      {&region, /*offset=*/0, /*length=*/8},
      {&region, /*offset=*/16, /*length=*/4},
  };
  struct ibv_sge sges[2];
  int sge_count = 0;

  IREE_ASSERT_OK(iree_net_rdma_sge_list_from_span_list(
      iree_async_span_list_make(spans, IREE_ARRAYSIZE(spans)),
      IREE_ARRAYSIZE(sges), sges, &sge_count));

  EXPECT_EQ(2, sge_count);
  EXPECT_EQ((uint64_t)(uintptr_t)storage, sges[0].addr);
  EXPECT_EQ(8u, sges[0].length);
  EXPECT_EQ((uint64_t)(uintptr_t)(storage + 16), sges[1].addr);
  EXPECT_EQ(4u, sges[1].length);
}

TEST(RdmaSgeTest, RejectsInvalidSpanList) {
  char storage[64] = {0};
  iree_async_region_t region = MakeRegion(storage, sizeof(storage));
  iree_async_span_t spans[2] = {
      {&region, /*offset=*/0, /*length=*/8},
      {nullptr, /*offset=*/0, /*length=*/4},
  };
  struct ibv_sge sges[2];
  int sge_count = 0;

  iree_async_span_list_t missing_values = {nullptr, 1};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_rdma_sge_list_from_span_list(
          missing_values, IREE_ARRAYSIZE(sges), sges, &sge_count));
  EXPECT_EQ(0, sge_count);

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_net_rdma_sge_list_from_span_list(
          iree_async_span_list_make(spans, IREE_ARRAYSIZE(spans)),
          /*sge_capacity=*/1, sges, &sge_count));
  EXPECT_EQ(0, sge_count);

  sges[0].addr = 1;
  sges[0].length = 2;
  sges[0].lkey = 3;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_rdma_sge_list_from_span_list(
          iree_async_span_list_make(spans, IREE_ARRAYSIZE(spans)),
          IREE_ARRAYSIZE(sges), sges, &sge_count));
  EXPECT_EQ(0u, sges[0].addr);
  EXPECT_EQ(0u, sges[0].length);
  EXPECT_EQ(0u, sges[0].lkey);
  EXPECT_EQ(0, sge_count);
}

TEST(RdmaSgeTest, AcceptsEmptySpanList) {
  int sge_count = 1;
  IREE_ASSERT_OK(iree_net_rdma_sge_list_from_span_list(
      iree_async_span_list_empty(), /*sge_capacity=*/0, nullptr, &sge_count));
  EXPECT_EQ(0, sge_count);
}

}  // namespace
