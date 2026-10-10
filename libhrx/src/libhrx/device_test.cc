// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hrx_internal.h"
#include "iree/hal/testing/mock_device.h"
#include "iree/hal/topology_builder.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

iree_hal_topology_edge_t MakePeerEdge(
    iree_hal_topology_interop_mode_t noncoherent_read,
    iree_hal_topology_interop_mode_t noncoherent_write,
    iree_hal_topology_interop_mode_t coherent_read,
    iree_hal_topology_interop_mode_t coherent_write,
    iree_hal_topology_capability_t capabilities) {
  iree_hal_topology_edge_t edge = iree_hal_topology_edge_empty();
  edge.lo = iree_hal_topology_edge_set_buffer_read_mode_noncoherent(
      edge.lo, noncoherent_read);
  edge.lo = iree_hal_topology_edge_set_buffer_write_mode_noncoherent(
      edge.lo, noncoherent_write);
  edge.lo = iree_hal_topology_edge_set_buffer_read_mode_coherent(edge.lo,
                                                                 coherent_read);
  edge.lo = iree_hal_topology_edge_set_buffer_write_mode_coherent(
      edge.lo, coherent_write);
  edge.lo = iree_hal_topology_edge_set_capability_flags(edge.lo, capabilities);
  return edge;
}

TEST(DevicePeerAccessTest, RequiresUsableNoncoherentReadAndWrite) {
  struct TestCase {
    const char* name;
    iree_hal_topology_interop_mode_t noncoherent_read;
    iree_hal_topology_interop_mode_t noncoherent_write;
    iree_hal_topology_interop_mode_t coherent_read;
    iree_hal_topology_interop_mode_t coherent_write;
    iree_hal_topology_capability_t capabilities;
    bool expected;
  };
  const TestCase test_cases[] = {
      {
          "native coarse read and write",
          IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE,
          IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE,
          IREE_HAL_TOPOLOGY_INTEROP_MODE_NONE,
          IREE_HAL_TOPOLOGY_INTEROP_MODE_NONE,
          IREE_HAL_TOPOLOGY_CAPABILITY_NONE,
          true,
      },
      {
          "native coarse access ignores an unrelated grant bit",
          IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE,
          IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE,
          IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY,
          IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY,
          IREE_HAL_TOPOLOGY_CAPABILITY_PEER_ACCESS_REQUIRES_GRANT,
          true,
      },
      {
          "grant-required coarse copy",
          IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY,
          IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY,
          IREE_HAL_TOPOLOGY_INTEROP_MODE_NONE,
          IREE_HAL_TOPOLOGY_INTEROP_MODE_NONE,
          IREE_HAL_TOPOLOGY_CAPABILITY_PEER_ACCESS_REQUIRES_GRANT,
          true,
      },
      {
          "fine native with coarse unavailable",
          IREE_HAL_TOPOLOGY_INTEROP_MODE_NONE,
          IREE_HAL_TOPOLOGY_INTEROP_MODE_NONE,
          IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE,
          IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE,
          IREE_HAL_TOPOLOGY_CAPABILITY_NONE,
          false,
      },
      {
          "fine native with coarse copy",
          IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY,
          IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY,
          IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE,
          IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE,
          IREE_HAL_TOPOLOGY_CAPABILITY_NONE,
          false,
      },
      {
          "grant-required coarse copy with fine native access",
          IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY,
          IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY,
          IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE,
          IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE,
          IREE_HAL_TOPOLOGY_CAPABILITY_PEER_ACCESS_REQUIRES_GRANT,
          true,
      },
      {
          "grant does not create unavailable coarse access",
          IREE_HAL_TOPOLOGY_INTEROP_MODE_NONE,
          IREE_HAL_TOPOLOGY_INTEROP_MODE_NONE,
          IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE,
          IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE,
          IREE_HAL_TOPOLOGY_CAPABILITY_PEER_ACCESS_REQUIRES_GRANT,
          false,
      },
      {
          "native coarse read without native write",
          IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE,
          IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY,
          IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE,
          IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE,
          IREE_HAL_TOPOLOGY_CAPABILITY_NONE,
          false,
      },
      {
          "native coarse write without native read",
          IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY,
          IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE,
          IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE,
          IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE,
          IREE_HAL_TOPOLOGY_CAPABILITY_NONE,
          false,
      },
  };

  for (const TestCase& test_case : test_cases) {
    SCOPED_TRACE(test_case.name);
    const iree_hal_topology_edge_t edge =
        MakePeerEdge(test_case.noncoherent_read, test_case.noncoherent_write,
                     test_case.coherent_read, test_case.coherent_write,
                     test_case.capabilities);
    EXPECT_EQ(hrx_topology_edge_supports_peer_access(edge), test_case.expected);
  }
}

class DevicePeerAccessIntegrationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_hal_mock_device_options_t options;
    iree_hal_mock_device_options_initialize(&options);
    options.identifier = iree_make_cstring_view("peer-0");
    IREE_ASSERT_OK(iree_hal_mock_device_create(
        &options, iree_allocator_system(), &hal_devices_[0]));
    options.identifier = iree_make_cstring_view("peer-1");
    IREE_ASSERT_OK(iree_hal_mock_device_create(
        &options, iree_allocator_system(), &hal_devices_[1]));

    iree_hal_topology_builder_t builder;
    iree_hal_topology_builder_initialize(&builder, 2);
    const iree_hal_topology_edge_t native_edge = MakePeerEdge(
        IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE,
        IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE,
        IREE_HAL_TOPOLOGY_INTEROP_MODE_NONE,
        IREE_HAL_TOPOLOGY_INTEROP_MODE_NONE, IREE_HAL_TOPOLOGY_CAPABILITY_NONE);
    IREE_ASSERT_OK(
        iree_hal_topology_builder_set_edge(&builder, 0, 1, native_edge));
    IREE_ASSERT_OK(
        iree_hal_topology_builder_set_edge(&builder, 1, 0, native_edge));
    IREE_ASSERT_OK(iree_hal_topology_builder_finalize(
        &builder, iree_allocator_system(), &topology_));

    for (uint32_t i = 0; i < 2; ++i) {
      iree_hal_device_topology_info_t topology_info = {
          .self_edge = iree_hal_topology_query_edge(topology_, i, i).lo,
          .topology_index = i,
          .topology = topology_,
      };
      IREE_ASSERT_OK(iree_hal_device_assign_topology_info(hal_devices_[i],
                                                          &topology_info));
      devices_[i].type = HRX_ACCELERATOR_GPU;
      devices_[i].hal_device = hal_devices_[i];
    }
  }

  void TearDown() override {
    for (iree_hal_device_t*& hal_device : hal_devices_) {
      if (hal_device) {
        IREE_EXPECT_OK(iree_hal_device_assign_topology_info(hal_device, NULL));
        iree_hal_device_release(hal_device);
        hal_device = nullptr;
      }
    }
    iree_hal_topology_destroy(topology_, iree_allocator_system());
  }

  iree_hal_device_t* hal_devices_[2] = {};
  hrx_device_s devices_[2] = {};
  iree_hal_topology_t* topology_ = nullptr;
};

TEST_F(DevicePeerAccessIntegrationTest,
       LowerQueryUsesNativeCoarseAccessWithoutMutatingTopology) {
  const iree_hal_topology_edge_t before_0_to_1 =
      iree_hal_topology_query_edge(topology_, 0, 1);
  const iree_hal_topology_edge_t before_1_to_0 =
      iree_hal_topology_query_edge(topology_, 1, 0);

  bool can_access = false;
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_device_can_access_peer(&devices_[0], &devices_[1], &can_access)));
  EXPECT_TRUE(can_access);

  can_access = false;
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_device_can_access_peer(&devices_[1], &devices_[0], &can_access)));
  EXPECT_TRUE(can_access);

  const iree_hal_topology_edge_t after_0_to_1 =
      iree_hal_topology_query_edge(topology_, 0, 1);
  const iree_hal_topology_edge_t after_1_to_0 =
      iree_hal_topology_query_edge(topology_, 1, 0);
  EXPECT_EQ(after_0_to_1.lo, before_0_to_1.lo);
  EXPECT_EQ(after_0_to_1.hi, before_0_to_1.hi);
  EXPECT_EQ(after_1_to_0.lo, before_1_to_0.lo);
  EXPECT_EQ(after_1_to_0.hi, before_1_to_0.hi);
}

TEST_F(DevicePeerAccessIntegrationTest, PreservesSameDeviceAndNullMappings) {
  bool can_access = false;
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_device_can_access_peer(&devices_[0], &devices_[0], &can_access)));
  EXPECT_TRUE(can_access);

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        hrx_status_to_iree(hrx_device_can_access_peer(
                            nullptr, &devices_[0], &can_access)));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        hrx_status_to_iree(hrx_device_can_access_peer(
                            &devices_[0], nullptr, &can_access)));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        hrx_status_to_iree(hrx_device_can_access_peer(
                            &devices_[0], &devices_[1], nullptr)));
}

}  // namespace
