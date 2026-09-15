// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/wddm/wkmi/loader.h"

#include <cstdint>
#include <vector>

#include "gtest/gtest.h"
#include "libamdf/src/allocator.h"

namespace {

using SetQueryResultFn = void(__cdecl*)(amdf_wkmi_bridge_result_t);

class WkmiLoaderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const DWORD path_capacity =
        GetEnvironmentVariableW(L"AMDF_WKMI_BRIDGE_PATH", nullptr, 0);
    ASSERT_GT(path_capacity, 0u);
    bridge_path_.resize(path_capacity);
    ASSERT_LT(GetEnvironmentVariableW(L"AMDF_WKMI_BRIDGE_PATH",
                                      bridge_path_.data(), path_capacity),
              path_capacity);
  }

  void TearDown() override {
    if (!bridge_path_.empty()) {
      EXPECT_TRUE(SetEnvironmentVariableW(L"AMDF_WKMI_BRIDGE_PATH",
                                          bridge_path_.data()));
    }
  }

  // Original fake-bridge path restored after every test.
  std::vector<wchar_t> bridge_path_;
};

TEST_F(WkmiLoaderTest, FailedInitializationLeavesOutputUnchanged) {
  std::vector<wchar_t> missing_path = bridge_path_;
  missing_path.pop_back();
  constexpr wchar_t kMissingSuffix[] = L".missing";
  missing_path.insert(
      missing_path.end(), kMissingSuffix,
      kMissingSuffix + sizeof(kMissingSuffix) / sizeof(kMissingSuffix[0]));
  ASSERT_TRUE(
      SetEnvironmentVariableW(L"AMDF_WKMI_BRIDGE_PATH", missing_path.data()));

  const HMODULE sentinel = reinterpret_cast<HMODULE>(uintptr_t{1});
  amdf_gpu_wddm_wkmi_loader_t loader = {};
  loader.module = sentinel;
  EXPECT_FALSE(amdf_status_is_ok(
      amdf_gpu_wddm_wkmi_loader_initialize(amdf_allocator_system(), &loader)));
  EXPECT_EQ(loader.module, sentinel);
}

TEST_F(WkmiLoaderTest, NegotiationFailurePreservesLiveModuleOwner) {
  amdf_gpu_wddm_wkmi_loader_t loader = {};
  ASSERT_EQ(
      amdf_gpu_wddm_wkmi_loader_initialize(amdf_allocator_system(), &loader),
      AMDF_STATUS_OK);
  ASSERT_NE(loader.module, nullptr);
  const HMODULE loaded_module = loader.module;

  const auto set_query_result = reinterpret_cast<SetQueryResultFn>(
      GetProcAddress(loader.module, "amdf_test_wkmi_bridge_set_query_result"));
  ASSERT_NE(set_query_result, nullptr);
  set_query_result(AMDF_WKMI_BRIDGE_RESULT_VERSION_MISMATCH);

  const auto* sentinel_api =
      reinterpret_cast<const amdf_wkmi_bridge_api_t*>(uintptr_t{1});
  const amdf_wkmi_bridge_api_t* api = sentinel_api;
  EXPECT_EQ(
      amdf_status_code(amdf_gpu_wddm_wkmi_loader_query_api(&loader, &api)),
      AMDF_STATUS_CODE_VERSION_MISMATCH);
  EXPECT_EQ(api, sentinel_api);
  EXPECT_EQ(loader.module, loaded_module);

  set_query_result(AMDF_WKMI_BRIDGE_RESULT_SUCCESS);
  ASSERT_EQ(amdf_gpu_wddm_wkmi_loader_query_api(&loader, &api), AMDF_STATUS_OK);
  EXPECT_NE(api, nullptr);
  EXPECT_EQ(loader.module, loaded_module);
  EXPECT_EQ(amdf_gpu_wddm_wkmi_loader_deinitialize(&loader), AMDF_STATUS_OK);
  EXPECT_EQ(loader.module, nullptr);
}

}  // namespace
