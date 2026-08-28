// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <dlfcn.h>
#include <limits.h>
#include <link.h>

#include <climits>
#include <cstdlib>
#include <string>

#include "api.h"
#include "iree/testing/gtest.h"

namespace {

const char* CandidateLibPath() {
  if (const char* env = std::getenv("HRX_TEST_LIBAMDHIP64");
      env && *env != '\0') {
    return env;
  }
#ifdef HRX_TEST_LIBAMDHIP64_PATH
  return HRX_TEST_LIBAMDHIP64_PATH;
#else
  return nullptr;
#endif
}

std::string CanonicalPath(const char* path) {
  char canonical_path[PATH_MAX];
  return path && realpath(path, canonical_path) ? canonical_path : "";
}

using HipInitFn = hipError_t (*)(unsigned int flags);
using HipHalDeinitFn = hipError_t (*)(void);
using HipGetDeviceFn = hipError_t (*)(int* device);
using HipSetDeviceFn = hipError_t (*)(int device);
using HipGetDeviceCountFn = hipError_t (*)(int* count);
using HipDeviceGetAttributeFn = hipError_t (*)(int* value,
                                               hipDeviceAttribute_t attribute,
                                               int device);
using HipGetLastErrorFn = hipError_t (*)(void);
using HipPeekAtLastErrorFn = hipError_t (*)(void);

// Resolves |name| from |library| and verifies that the dynamic loader bound it
// to the exact DSO named by |expected_library_path|.
template <typename T>
T ResolveCandidateSymbol(void* library, const char* name,
                         const std::string& expected_library_path) {
  dlerror();
  void* symbol = dlsym(library, name);
  const char* error = dlerror();
  EXPECT_EQ(nullptr, error) << name << ": " << (error ? error : "");
  EXPECT_NE(nullptr, symbol) << name;
  if (!symbol) return nullptr;

  Dl_info symbol_info = {};
  EXPECT_NE(0, dladdr(symbol, &symbol_info)) << name;
  if (!symbol_info.dli_fname) return reinterpret_cast<T>(symbol);
  EXPECT_EQ(expected_library_path, CanonicalPath(symbol_info.dli_fname))
      << name << " resolved from " << symbol_info.dli_fname;
  return reinterpret_cast<T>(symbol);
}

// Owns the RTLD_LOCAL handle and exact public entry points exercised by this
// test.
struct HipRuntimeApi {
  // Handle returned by dlopen for the HIP runtime instance under test.
  void* library = nullptr;
  // Canonical filesystem path of the loaded HIP runtime instance.
  std::string library_path;
  // Initializes the loaded HIP runtime instance.
  HipInitFn init = nullptr;
  // Deinitializes the loaded HRX runtime instance before unloading its DSO.
  HipHalDeinitFn hal_deinit = nullptr;
  // Reports the calling thread's current device.
  HipGetDeviceFn get_device = nullptr;
  // Selects the calling thread's current device.
  HipSetDeviceFn set_device = nullptr;
  // Reports every device visible to the loaded runtime instance.
  HipGetDeviceCountFn get_device_count = nullptr;
  // Reports one public device attribute for one visible device.
  HipDeviceGetAttributeFn device_get_attribute = nullptr;
  // Returns and clears the calling thread's last HIP error.
  HipGetLastErrorFn get_last_error = nullptr;
  // Returns without clearing the calling thread's last HIP error.
  HipPeekAtLastErrorFn peek_at_last_error = nullptr;
};

class HipUnifiedAddressingAttributeApiTest : public testing::Test {
 protected:
  void SetUp() override {
    if (!api_.library) {
      const char* library_path = CandidateLibPath();
      ASSERT_NE(nullptr, library_path)
          << "the build must provide the libamdhip64 artifact under test";
      api_.library_path = CanonicalPath(library_path);
      ASSERT_FALSE(api_.library_path.empty())
          << "cannot canonicalize " << library_path;
      api_.library = dlopen(library_path, RTLD_NOW | RTLD_LOCAL);
      ASSERT_NE(nullptr, api_.library)
          << "cannot dlopen " << library_path << ": " << dlerror();

      link_map* loaded_map = nullptr;
      ASSERT_EQ(0, dlinfo(api_.library, RTLD_DI_LINKMAP, &loaded_map));
      ASSERT_NE(nullptr, loaded_map);
      ASSERT_EQ(api_.library_path, CanonicalPath(loaded_map->l_name));

      api_.init = ResolveCandidateSymbol<HipInitFn>(api_.library, "hipInit",
                                                    api_.library_path);
      api_.hal_deinit = ResolveCandidateSymbol<HipHalDeinitFn>(
          api_.library, "hipHALDeinit", api_.library_path);
      api_.get_device = ResolveCandidateSymbol<HipGetDeviceFn>(
          api_.library, "hipGetDevice", api_.library_path);
      api_.set_device = ResolveCandidateSymbol<HipSetDeviceFn>(
          api_.library, "hipSetDevice", api_.library_path);
      api_.get_device_count = ResolveCandidateSymbol<HipGetDeviceCountFn>(
          api_.library, "hipGetDeviceCount", api_.library_path);
      api_.device_get_attribute =
          ResolveCandidateSymbol<HipDeviceGetAttributeFn>(
              api_.library, "hipDeviceGetAttribute", api_.library_path);
      api_.get_last_error = ResolveCandidateSymbol<HipGetLastErrorFn>(
          api_.library, "hipGetLastError", api_.library_path);
      api_.peek_at_last_error = ResolveCandidateSymbol<HipPeekAtLastErrorFn>(
          api_.library, "hipPeekAtLastError", api_.library_path);
    }

    ASSERT_NE(nullptr, api_.init);
    ASSERT_NE(nullptr, api_.hal_deinit);
    ASSERT_NE(nullptr, api_.get_device);
    ASSERT_NE(nullptr, api_.set_device);
    ASSERT_NE(nullptr, api_.get_device_count);
    ASSERT_NE(nullptr, api_.device_get_attribute);
    ASSERT_NE(nullptr, api_.get_last_error);
    ASSERT_NE(nullptr, api_.peek_at_last_error);

    const hipError_t init_result = api_.init(/*flags=*/0);
    if (init_result == hipErrorNoDevice) {
      GTEST_SKIP() << "the loaded runtime reports no compatible devices";
    }
    ASSERT_EQ(hipSuccess, init_result);
    ASSERT_EQ(hipSuccess, api_.get_device_count(&device_count_));
    if (device_count_ == 0) {
      GTEST_SKIP() << "the loaded runtime reports zero compatible devices";
    }
    ASSERT_GT(device_count_, 0);
    ASSERT_EQ(hipSuccess, api_.get_last_error());
    current_device_ = device_count_ - 1;
    ASSERT_EQ(hipSuccess, api_.set_device(current_device_));
    ExpectCurrentDevice();
    ASSERT_EQ(hipSuccess, api_.peek_at_last_error());
  }

  static void TearDownTestSuite() {
    if (!api_.library) return;
    void* library = api_.library;
    ASSERT_NE(nullptr, api_.hal_deinit);
    EXPECT_EQ(hipSuccess, api_.hal_deinit());
    api_ = {};
    EXPECT_EQ(0, dlclose(library));
  }

  void ExpectCurrentDevice() {
    int actual_device = -1;
    ASSERT_EQ(hipSuccess, api_.get_device(&actual_device));
    EXPECT_EQ(current_device_, actual_device);
  }

  template <typename Callable>
  void ExpectErrorState(hipError_t expected, Callable&& callable) {
    ASSERT_EQ(hipSuccess, api_.get_last_error());
    EXPECT_EQ(expected, callable());
    EXPECT_EQ(expected, api_.peek_at_last_error());
    ExpectCurrentDevice();
    EXPECT_EQ(expected, api_.peek_at_last_error());
    EXPECT_EQ(expected, api_.get_last_error());
    EXPECT_EQ(hipSuccess, api_.peek_at_last_error());
  }

  // Process-lifetime runtime instance loaded from the exact built DSO.
  static HipRuntimeApi api_;
  // Number of compatible devices visible to the loaded runtime.
  int device_count_ = 0;
  // Device selected in calling-thread TLS before each attribute query.
  int current_device_ = -1;
};

HipRuntimeApi HipUnifiedAddressingAttributeApiTest::api_;

TEST_F(HipUnifiedAddressingAttributeApiTest, ReturnsOneOnEveryVisibleDevice) {
  for (int device = 0; device < device_count_; ++device) {
    SCOPED_TRACE(device);
    int attribute_value = -1;
    ASSERT_EQ(hipSuccess, api_.device_get_attribute(
                              &attribute_value,
                              hipDeviceAttributeUnifiedAddressing, device));
    EXPECT_EQ(1, attribute_value);
    ExpectCurrentDevice();
    EXPECT_EQ(hipSuccess, api_.peek_at_last_error());
    EXPECT_EQ(hipSuccess, api_.get_last_error());
  }
}

TEST_F(HipUnifiedAddressingAttributeApiTest,
       NullOutputTakesPrecedenceAndPublishesLastError) {
  const int device_ids[] = {0, -1, device_count_, INT_MIN, INT_MAX};
  for (int device : device_ids) {
    SCOPED_TRACE(device);
    ExpectErrorState(hipErrorInvalidValue, [&] {
      return api_.device_get_attribute(
          nullptr, hipDeviceAttributeUnifiedAddressing, device);
    });
  }
}

TEST_F(HipUnifiedAddressingAttributeApiTest,
       InvalidDevicesPreserveOutputAndPublishLastError) {
  const int invalid_devices[] = {-1, device_count_, INT_MIN, INT_MAX};
  for (int device : invalid_devices) {
    SCOPED_TRACE(device);
    int attribute_value = 0x5a5a5a5a;
    ExpectErrorState(hipErrorInvalidDevice, [&] {
      return api_.device_get_attribute(
          &attribute_value, hipDeviceAttributeUnifiedAddressing, device);
    });
    EXPECT_EQ(0x5a5a5a5a, attribute_value);
  }
}

TEST_F(HipUnifiedAddressingAttributeApiTest, SuccessPreservesPendingLastError) {
  ASSERT_EQ(hipSuccess, api_.get_last_error());
  ASSERT_EQ(hipErrorInvalidValue,
            api_.device_get_attribute(
                nullptr, hipDeviceAttributeUnifiedAddressing, /*device=*/0));
  int attribute_value = -1;
  EXPECT_EQ(hipSuccess,
            api_.device_get_attribute(&attribute_value,
                                      hipDeviceAttributeUnifiedAddressing,
                                      /*device=*/0));
  EXPECT_EQ(1, attribute_value);
  ExpectCurrentDevice();
  EXPECT_EQ(hipErrorInvalidValue, api_.peek_at_last_error());
  EXPECT_EQ(hipErrorInvalidValue, api_.get_last_error());
  EXPECT_EQ(hipSuccess, api_.peek_at_last_error());
}

}  // namespace
