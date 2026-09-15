// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/wddm/wkmi/adapter.h"

#include <cstdint>
#include <cstring>

#include "gtest/gtest.h"
#include "libamdf/src/allocator.h"

namespace {

using ResetBridgeFn = void(__cdecl*)(void);
using SetQueryResultFn = void(__cdecl*)(amdf_wkmi_bridge_result_t);

TEST(WkmiAdapterTest, FailedInitializationLeavesOutputsUnchanged) {
  amdf_gpu_wddm_wkmi_loader_t loader = {};
  const amdf_allocator_t host_allocator = amdf_allocator_system();
  ASSERT_EQ(amdf_gpu_wddm_wkmi_loader_initialize(host_allocator, &loader),
            AMDF_STATUS_OK);

  const auto reset_bridge = reinterpret_cast<ResetBridgeFn>(
      GetProcAddress(loader.module, "amdf_test_wkmi_bridge_reset"));
  const auto set_query_result = reinterpret_cast<SetQueryResultFn>(
      GetProcAddress(loader.module, "amdf_test_wkmi_bridge_set_query_result"));
  ASSERT_NE(reset_bridge, nullptr);
  ASSERT_NE(set_query_result, nullptr);
  reset_bridge();
  set_query_result(AMDF_WKMI_BRIDGE_RESULT_VERSION_MISMATCH);

  const amdf_gpu_wddm_wkmi_adapter_t sentinel_adapter = {
      reinterpret_cast<const amdf_wkmi_bridge_api_t*>(uintptr_t{1}),
      reinterpret_cast<amdf_wkmi_bridge_gpu_adapter_t*>(uintptr_t{2}),
  };
  amdf_gpu_wddm_wkmi_adapter_t adapter = sentinel_adapter;
  amdf_wkmi_bridge_gpu_properties_t properties = {};
  properties.gfx_ip_major = 99;
  const amdf_wkmi_bridge_gpu_properties_t sentinel_properties = properties;

  EXPECT_EQ(amdf_status_code(amdf_gpu_wddm_wkmi_adapter_initialize(
                &loader, 0x08, 0, host_allocator, &adapter, &properties)),
            AMDF_STATUS_CODE_VERSION_MISMATCH);
  EXPECT_EQ(adapter.api, sentinel_adapter.api);
  EXPECT_EQ(adapter.native, sentinel_adapter.native);
  EXPECT_EQ(std::memcmp(&properties, &sentinel_properties, sizeof(properties)),
            0);

  set_query_result(AMDF_WKMI_BRIDGE_RESULT_SUCCESS);
  adapter = {};
  properties = {};
  ASSERT_EQ(amdf_gpu_wddm_wkmi_adapter_initialize(
                &loader, 0x08, 0, host_allocator, &adapter, &properties),
            AMDF_STATUS_OK);
  EXPECT_NE(adapter.api, nullptr);
  EXPECT_NE(adapter.native, nullptr);
  EXPECT_EQ(properties.gfx_ip_major, 11);
  EXPECT_EQ(properties.gfx_ip_minor, 5);

  EXPECT_EQ(amdf_status_code(amdf_gpu_wddm_wkmi_adapter_deinitialize(&adapter)),
            AMDF_STATUS_CODE_BUSY);
  EXPECT_NE(adapter.api, nullptr);
  EXPECT_NE(adapter.native, nullptr);
  EXPECT_EQ(amdf_gpu_wddm_wkmi_adapter_deinitialize(&adapter), AMDF_STATUS_OK);
  EXPECT_EQ(adapter.api, nullptr);
  EXPECT_EQ(adapter.native, nullptr);
  EXPECT_EQ(amdf_gpu_wddm_wkmi_loader_deinitialize(&loader), AMDF_STATUS_OK);
}

}  // namespace
