// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <dlfcn.h>

#include <climits>
#include <cstdlib>

#include "api.h"
#include "hip_device_attribute_validation_test_shim.h"
#include "iree/testing/gtest.h"

namespace {

constexpr int kOutputSentinel = 0x5a5a5a5a;

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

using HipDeviceGetAttributeFn = hipError_t (*)(int* value,
                                               hipDeviceAttribute_t attribute,
                                               int device);
using HipGetLastErrorFn = hipError_t (*)(void);
using HipPeekAtLastErrorFn = hipError_t (*)(void);
using HipHalDeinitFn = hipError_t (*)(void);

template <typename T>
T ResolveCandidateSymbol(void* library, const char* name) {
  dlerror();
  void* symbol = dlsym(library, name);
  const char* error = dlerror();
  EXPECT_EQ(nullptr, error) << name << ": " << (error ? error : "");
  EXPECT_NE(nullptr, symbol) << name;
  return reinterpret_cast<T>(symbol);
}

TEST(HipDeviceAttributeNoDeviceApiTest,
     InitializationPrecedesArgumentValidation) {
  ASSERT_STREQ("-1", std::getenv("ROCR_VISIBLE_DEVICES"));
  const char* library_path = CandidateLibPath();
  ASSERT_NE(nullptr, library_path)
      << "the build must provide the libamdhip64 artifact under test";
  void* library = dlopen(library_path, RTLD_NOW | RTLD_LOCAL);
  ASSERT_NE(nullptr, library)
      << "cannot dlopen " << library_path << ": " << dlerror();

  HipDeviceGetAttributeFn device_get_attribute =
      ResolveCandidateSymbol<HipDeviceGetAttributeFn>(library,
                                                      "hipDeviceGetAttribute");
  HipGetLastErrorFn get_last_error =
      ResolveCandidateSymbol<HipGetLastErrorFn>(library, "hipGetLastError");
  HipPeekAtLastErrorFn peek_at_last_error =
      ResolveCandidateSymbol<HipPeekAtLastErrorFn>(library,
                                                   "hipPeekAtLastError");
  HipHalDeinitFn hal_deinit =
      ResolveCandidateSymbol<HipHalDeinitFn>(library, "hipHALDeinit");
  ASSERT_NE(nullptr, device_get_attribute);
  ASSERT_NE(nullptr, get_last_error);
  ASSERT_NE(nullptr, peek_at_last_error);
  ASSERT_NE(nullptr, hal_deinit);

  EXPECT_EQ(hipErrorNoDevice,
            hrx_test_hip_device_get_attribute(
                device_get_attribute, nullptr,
                hipDeviceAttributeMaxThreadsPerBlock, /*device=*/0));
  EXPECT_EQ(hipErrorNoDevice, peek_at_last_error());
  EXPECT_EQ(hipErrorNoDevice, get_last_error());
  EXPECT_EQ(hipSuccess, peek_at_last_error());

  int attribute_value = kOutputSentinel;
  EXPECT_EQ(hipErrorNoDevice,
            hrx_test_hip_device_get_attribute(
                device_get_attribute, &attribute_value, INT_MAX, INT_MAX));
  EXPECT_EQ(kOutputSentinel, attribute_value);
  EXPECT_EQ(hipErrorNoDevice, peek_at_last_error());
  EXPECT_EQ(hipErrorNoDevice, get_last_error());
  EXPECT_EQ(hipSuccess, peek_at_last_error());

  EXPECT_EQ(hipSuccess, hal_deinit());
  EXPECT_EQ(0, dlclose(library));
}

}  // namespace
