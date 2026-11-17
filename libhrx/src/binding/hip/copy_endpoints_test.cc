// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "binding/hip/copy_endpoints.h"

#include <array>

#include "common/context.h"
#include "common/hrx_bridge.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "libhrx/src/libhrx/buffer_table.h"

namespace {

// Endpoint preparation consumes allocation metadata, not native queues or
// buffers. Populate the real pointer table with caller-owned metadata; public
// graph tests cover native execution and foreign-context ownership.
class CopyEndpointsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    hrx_buffer_table_initialize(&context_.buffer_table);
    for (size_t i = 0; i < allocations_.size(); ++i) {
      auto& allocation = allocations_[i];
      allocation.context = &context_;
      allocation.size = storage_[i].size();
      allocation.logical_size = i == 0 ? 12 : allocation.size;
      allocation.memory_kind = i == 0 ? IREE_HAL_STREAMING_MEMORY_KIND_DEVICE
                                      : IREE_HAL_STREAMING_MEMORY_KIND_HOST;
      IREE_ASSERT_OK(HRX_CALL(hrx_buffer_table_insert(
          &context_.buffer_table, (uintptr_t)storage_[i].data(),
          storage_[i].data(), storage_[i].size(), nullptr, &allocation)));
    }
  }

  void TearDown() override {
    hrx_buffer_table_deinitialize(&context_.buffer_table);
  }

  // Local pointer-table owner, borrowed throughout preparation.
  iree_hal_streaming_context_t context_ = {};
  // Addressable ranges standing for device and registered-host allocations.
  std::array<std::array<uint8_t, 16>, 2> storage_ = {};
  // Metadata kept alive by the caller, independent of endpoint preparation.
  std::array<iree_hal_streaming_buffer_t, 2> allocations_ = {};
};

TEST_F(CopyEndpointsTest, ClassifiesAndCarriesLocalReferences) {
  for (size_t source = 0; source < 2; ++source) {
    for (size_t target = 0; target < 2; ++target) {
      const hipMemcpyKind expected =
          source == 0
              ? (target == 0 ? hipMemcpyDeviceToDevice : hipMemcpyDeviceToHost)
              : (target == 0 ? hipMemcpyHostToDevice : hipMemcpyHostToHost);
      iree_hip_copy_endpoints_t endpoints;
      ASSERT_EQ(hipSuccess,
                iree_hip_copy_endpoints_initialize(
                    &context_, hipMemcpyDefault, storage_[target].data() + 2, 4,
                    storage_[source].data() + 1, 4, &endpoints));
      EXPECT_EQ(expected, endpoints.kind);
      EXPECT_EQ(&allocations_[source], endpoints.source.reference.buffer);
      EXPECT_EQ(1u, endpoints.source.reference.offset);
      EXPECT_EQ(&context_, endpoints.source.context);
      EXPECT_EQ(&allocations_[target], endpoints.target.reference.buffer);
      EXPECT_EQ(2u, endpoints.target.reference.offset);
      EXPECT_EQ(&context_, endpoints.target.context);
      iree_hip_copy_endpoints_deinitialize(&context_, &endpoints);
    }
  }
}

TEST_F(CopyEndpointsTest, DeviceBoundsUseTheRequestedAllocationExtent) {
  for (hipMemcpyKind kind : {hipMemcpyDeviceToDevice, hipMemcpyDefault,
                             hipMemcpyDeviceToDeviceNoCU}) {
    iree_hip_copy_endpoints_t endpoints;
    ASSERT_EQ(hipSuccess, iree_hip_copy_endpoints_initialize(
                              &context_, kind, storage_[0].data() + 10, 2,
                              storage_[0].data(), 2, &endpoints));
    EXPECT_EQ(hipMemcpyDeviceToDevice, endpoints.kind);
    iree_hip_copy_endpoints_deinitialize(&context_, &endpoints);
    EXPECT_EQ(hipErrorInvalidValue,
              iree_hip_copy_endpoints_initialize(
                  &context_, kind, storage_[0].data() + 10, 3,
                  storage_[0].data(), 2, &endpoints));
    EXPECT_EQ(hipErrorInvalidValue,
              iree_hip_copy_endpoints_initialize(
                  &context_, kind, storage_[0].data(), 2,
                  storage_[0].data() + 10, 3, &endpoints));
  }
}

TEST_F(CopyEndpointsTest, RegisteredHostOverrunNeverBecomesPageable) {
  std::array<uint8_t, 16> pageable = {};
  for (bool upload : {false, true}) {
    for (bool registered : {false, true}) {
      void* host = registered ? storage_[1].data() + 8 : pageable.data();
      void* device = storage_[0].data();
      const hipMemcpyKind kind =
          upload ? hipMemcpyHostToDevice : hipMemcpyDeviceToHost;
      iree_hip_copy_endpoints_t endpoints;
      const hipError_t result = iree_hip_copy_endpoints_initialize(
          &context_, kind, upload ? device : host, 9, upload ? host : device, 9,
          &endpoints);
      if (registered) {
        EXPECT_EQ(hipErrorInvalidValue, result);
      } else {
        ASSERT_EQ(hipSuccess, result);
        const auto& host_endpoint =
            upload ? endpoints.source : endpoints.target;
        EXPECT_EQ(nullptr, host_endpoint.reference.buffer);
        EXPECT_EQ(nullptr, host_endpoint.context);
        iree_hip_copy_endpoints_deinitialize(&context_, &endpoints);
      }
    }
  }
}

}  // namespace
