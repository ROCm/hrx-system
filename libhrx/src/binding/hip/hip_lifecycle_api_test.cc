// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <dlfcn.h>

#include <cstdlib>

#include "binding/hip/api.h"
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
  return "libamdhip64.so";
#endif
}

template <typename T>
T ResolveHipSymbol(void* library, const char* name) {
  return reinterpret_cast<T>(dlsym(library, name));
}

using HipInitFn = hipError_t (*)(unsigned int flags);
using HipEventCreateFn = hipError_t (*)(hipEvent_t* event);
using HipEventDestroyFn = hipError_t (*)(hipEvent_t event);
using HipHalDeinitFn = hipError_t (*)(void);

TEST(HipLifecycleApiTest, DeinitializeAfterEventLifecycle) {
  void* library = dlopen(CandidateLibPath(), RTLD_LAZY | RTLD_LOCAL);
  ASSERT_NE(library, nullptr)
      << "cannot dlopen " << CandidateLibPath() << ": " << dlerror();

  const auto init = ResolveHipSymbol<HipInitFn>(library, "hipInit");
  const auto event_create =
      ResolveHipSymbol<HipEventCreateFn>(library, "hipEventCreate");
  const auto event_destroy =
      ResolveHipSymbol<HipEventDestroyFn>(library, "hipEventDestroy");
  const auto hal_deinit =
      ResolveHipSymbol<HipHalDeinitFn>(library, "hipHALDeinit");
  ASSERT_NE(init, nullptr);
  ASSERT_NE(event_create, nullptr);
  ASSERT_NE(event_destroy, nullptr);
  ASSERT_NE(hal_deinit, nullptr);

  const hipError_t init_result = init(/*flags=*/0);
  if (init_result != hipSuccess) {
    EXPECT_EQ(0, dlclose(library));
    GTEST_SKIP() << "hipInit failed: " << init_result;
  }

  hipEvent_t event = nullptr;
  const hipError_t create_result = event_create(&event);
  EXPECT_EQ(hipSuccess, create_result);
  if (create_result == hipSuccess) {
    EXPECT_NE(event, nullptr);
    EXPECT_EQ(hipSuccess, event_destroy(event));
  }

  EXPECT_EQ(hipSuccess, hal_deinit());
  EXPECT_EQ(0, dlclose(library));
}

}  // namespace
