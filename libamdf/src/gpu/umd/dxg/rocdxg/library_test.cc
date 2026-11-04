// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <dlfcn.h>

#include <cstdlib>
#include <cstring>
#include <string>

#include "gtest/gtest.h"
#include "libamdf/src/allocator.h"
#include "libamdf/src/gpu/umd/dxg/bridge_api.h"

namespace {

bool ModuleIsMapped(const char* path) {
  void* module = dlopen(path, RTLD_NOW | RTLD_NOLOAD);
  if (module == nullptr) {
    return false;
  }
  dlclose(module);
  return true;
}

bool ExerciseSidecar(const char* path, std::string* out_error) {
  void* module = dlopen(path, RTLD_NOW | RTLD_LOCAL);
  if (module == nullptr) {
    *out_error = dlerror();
    return false;
  }

  void* query_symbol = dlsym(module, "amdf_dxg_bridge_query_api");
  amdf_dxg_bridge_query_api_fn_t query_api = nullptr;
  static_assert(sizeof(query_api) == sizeof(query_symbol));
  std::memcpy(&query_api, &query_symbol, sizeof(query_api));
  if (query_api == nullptr) {
    *out_error = "query symbol is absent";
    dlclose(module);
    return false;
  }

  const amdf_dxg_bridge_api_t* api = nullptr;
  if (query_api(AMDF_DXG_BRIDGE_ABI_VERSION_LATEST,
                AMDF_DXG_BRIDGE_ABI_VERSION_LATEST,
                &api) != AMDF_DXG_BRIDGE_RESULT_SUCCESS ||
      api == nullptr || api->structure_size != sizeof(*api) ||
      api->abi_version != AMDF_DXG_BRIDGE_ABI_VERSION_LATEST ||
      api->gpu_adapter_open == nullptr || api->gpu_adapter_close == nullptr) {
    *out_error = "query returned an incomplete bridge table";
    dlclose(module);
    return false;
  }
  if (api->gpu_adapter_open(nullptr, nullptr, nullptr, nullptr, nullptr) !=
      AMDF_DXG_BRIDGE_RESULT_INVALID_ARGUMENT) {
    *out_error = "invalid adapter open was accepted";
    dlclose(module);
    return false;
  }
  const amdf_dxg_bridge_gpu_adapter_open_info_t open_info = {
      .structure_size = sizeof(amdf_dxg_bridge_gpu_adapter_open_info_t),
      .adapter_handle = 1,
      .device_handle = 1,
  };
  const amdf_allocator_t host_allocator = amdf_allocator_system();
  amdf_dxg_bridge_gpu_adapter_t* adapter = nullptr;
  amdf_dxg_bridge_gpu_properties_t properties = {};
  uint32_t native_status = 0;
  if (api->gpu_adapter_open(&open_info, &host_allocator, &adapter, &properties,
                            &native_status) !=
      AMDF_DXG_BRIDGE_RESULT_INVALID_ARGUMENT) {
    *out_error = "adapter open accepted a missing native transport";
    dlclose(module);
    return false;
  }
  if (dlclose(module) != 0) {
    *out_error = dlerror();
    return false;
  }
  return true;
}

TEST(RocdxgLibraryTest, QueryIsSilentAndEveryMappingIsReleased) {
  const char* sidecar_path = std::getenv("AMDF_ROCDXG_BRIDGE_TEST_PATH");
  ASSERT_NE(sidecar_path, nullptr);

  for (int cycle = 0; cycle < 16; ++cycle) {
    std::string error;
    ::testing::internal::CaptureStdout();
    ::testing::internal::CaptureStderr();
    const bool succeeded = ExerciseSidecar(sidecar_path, &error);
    const std::string stderr_output = ::testing::internal::GetCapturedStderr();
    const std::string stdout_output = ::testing::internal::GetCapturedStdout();

    EXPECT_TRUE(succeeded) << error;
    EXPECT_TRUE(stdout_output.empty()) << stdout_output;
    EXPECT_TRUE(stderr_output.empty()) << stderr_output;
    EXPECT_FALSE(ModuleIsMapped(sidecar_path));
  }
}

}  // namespace
