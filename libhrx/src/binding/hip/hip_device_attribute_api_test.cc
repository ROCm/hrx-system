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

#include <cstdlib>
#include <string>

#include "api.h"
#include "iree/testing/gtest.h"

namespace {

constexpr int kOutputSentinel = 0x5a5a5a5a;

using HipDeviceGetAttributeFn = hipError_t (*)(int* value,
                                               hipDeviceAttribute_t attribute,
                                               int device);

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

TEST(HipDeviceAttributeApiTest, ReportsOnlyImplementedAttributes) {
  const char* library_path = CandidateLibPath();
  ASSERT_NE(nullptr, library_path)
      << "the build must provide the libamdhip64 artifact under test";
  const std::string canonical_library_path = CanonicalPath(library_path);
  ASSERT_FALSE(canonical_library_path.empty())
      << "cannot canonicalize " << library_path;

  void* library = dlopen(library_path, RTLD_NOW | RTLD_LOCAL);
  ASSERT_NE(nullptr, library)
      << "cannot dlopen " << library_path << ": " << dlerror();
  void* symbol = dlsym(library, "hipDeviceGetAttribute");
  ASSERT_NE(nullptr, symbol) << "hipDeviceGetAttribute is not exported";

  Dl_info symbol_info = {};
  ASSERT_NE(0, dladdr(symbol, &symbol_info));
  ASSERT_NE(nullptr, symbol_info.dli_fname);
  EXPECT_EQ(canonical_library_path, CanonicalPath(symbol_info.dli_fname));
  auto device_get_attribute = reinterpret_cast<HipDeviceGetAttributeFn>(symbol);

  int value = kOutputSentinel;
  EXPECT_EQ(hipSuccess, device_get_attribute(
                            &value, hipDeviceAttributeAccessPolicyMaxWindowSize,
                            /*device=*/0));
  EXPECT_EQ(0, value);

  value = kOutputSentinel;
  EXPECT_EQ(hipErrorInvalidValue,
            device_get_attribute(&value, hipDeviceAttributeUnused1,
                                 /*device=*/0));
  EXPECT_EQ(kOutputSentinel, value);

  // The initialized runtime owns process-lifetime driver services, so keep its
  // shared object loaded until test process exit.
}

}  // namespace
