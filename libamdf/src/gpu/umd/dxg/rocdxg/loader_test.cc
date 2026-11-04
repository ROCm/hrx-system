// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/dxg/rocdxg/loader.h"

#include <dlfcn.h>

#include <cstdint>
#include <cstdlib>
#include <string>

#include "gtest/gtest.h"
#include "libamdf/src/allocator.h"

namespace {

bool ModuleIsMapped(const char* path) {
  void* module = dlopen(path, RTLD_NOW | RTLD_NOLOAD);
  if (module == nullptr) {
    return false;
  }
  return dlclose(module) == 0;
}

class RocdxgLoaderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const char* bridge_path = std::getenv("AMDF_DXG_BRIDGE_PATH");
    ASSERT_NE(bridge_path, nullptr);
    bridge_path_ = bridge_path;
  }

  void TearDown() override {
    ASSERT_EQ(::setenv("AMDF_DXG_BRIDGE_PATH", bridge_path_.c_str(), 1), 0);
  }

  // Original bridge path restored after every test.
  std::string bridge_path_;
};

TEST_F(RocdxgLoaderTest, LoadsQueriesAndUnloadsBridge) {
  EXPECT_FALSE(ModuleIsMapped(bridge_path_.c_str()));

  amdf_gpu_dxg_rocdxg_loader_t loader = {};
  ASSERT_EQ(
      amdf_gpu_dxg_rocdxg_loader_initialize(amdf_allocator_system(), &loader),
      AMDF_STATUS_OK);
  ASSERT_NE(loader.module, nullptr);
  EXPECT_TRUE(ModuleIsMapped(bridge_path_.c_str()));

  const amdf_dxg_bridge_api_t* api = nullptr;
  ASSERT_EQ(amdf_gpu_dxg_rocdxg_loader_query_api(&loader, &api),
            AMDF_STATUS_OK);
  ASSERT_NE(api, nullptr);
  EXPECT_EQ(api->abi_version, AMDF_DXG_BRIDGE_ABI_VERSION_13);

  amdf_gpu_dxg_rocdxg_loader_deinitialize(&loader);
  EXPECT_EQ(loader.module, nullptr);
  EXPECT_FALSE(ModuleIsMapped(bridge_path_.c_str()));
}

TEST_F(RocdxgLoaderTest, MissingBridgeLeavesOutputUnchanged) {
  const std::string missing_path = bridge_path_ + ".missing";
  ASSERT_EQ(::setenv("AMDF_DXG_BRIDGE_PATH", missing_path.c_str(), 1), 0);

  void* const sentinel = reinterpret_cast<void*>(uintptr_t{1});
  amdf_gpu_dxg_rocdxg_loader_t loader = {.module = sentinel};
  EXPECT_FALSE(amdf_status_is_ok(
      amdf_gpu_dxg_rocdxg_loader_initialize(amdf_allocator_system(), &loader)));
  EXPECT_EQ(loader.module, sentinel);
}

TEST_F(RocdxgLoaderTest, EmptyOverrideLeavesOutputUnchanged) {
  ASSERT_EQ(::setenv("AMDF_DXG_BRIDGE_PATH", "", 1), 0);

  void* const sentinel = reinterpret_cast<void*>(uintptr_t{1});
  amdf_gpu_dxg_rocdxg_loader_t loader = {.module = sentinel};
  EXPECT_EQ(
      amdf_gpu_dxg_rocdxg_loader_initialize(amdf_allocator_system(), &loader),
      amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT));
  EXPECT_EQ(loader.module, sentinel);
}

TEST_F(RocdxgLoaderTest, LoadsBridgeBesideOwningModule) {
  ASSERT_EQ(::unsetenv("AMDF_DXG_BRIDGE_PATH"), 0);

  amdf_gpu_dxg_rocdxg_loader_t loader = {};
  ASSERT_EQ(
      amdf_gpu_dxg_rocdxg_loader_initialize(amdf_allocator_system(), &loader),
      AMDF_STATUS_OK);
  const amdf_dxg_bridge_api_t* api = nullptr;
  ASSERT_EQ(amdf_gpu_dxg_rocdxg_loader_query_api(&loader, &api),
            AMDF_STATUS_OK);
  amdf_gpu_dxg_rocdxg_loader_deinitialize(&loader);
}

}  // namespace
