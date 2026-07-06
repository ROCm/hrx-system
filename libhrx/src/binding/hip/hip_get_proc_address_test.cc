// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

// Regression test for hipGetProcAddress() symbol resolution scope.
//
// hipGetProcAddress() must resolve symbols against the HIP runtime library
// that defines them, NOT the process-global symbol scope. Consumers such as
// Triton's AMD backend dlopen() libamdhip64 with RTLD_LOCAL and then resolve
// the whole HIP API through hipGetProcAddress(). When hipGetProcAddress() was
// implemented as dlsym(dlopen(NULL), symbol) it searched only the global
// scope, so an RTLD_LOCAL load left our symbols invisible and every lookup
// (starting with the very first, hipGetLastError) failed with
// hipErrorNotFound -- taking down Triton at driver-init time. See issue for
// details.
//
// The library under test is located via, in order: the HRX_TEST_LIBAMDHIP64
// environment variable (an explicit override), then the
// HRX_TEST_LIBAMDHIP64_PATH compile definition (set by CMake to the freshly
// built artifact), then the bare soname "libamdhip64.so" (resolved through the
// loader search path). If none can be opened the test is skipped rather than
// failed, so it is inert in build configurations that do not produce the
// shared object.

#include <dlfcn.h>

#include <cstdint>
#include <cstdlib>

#include "iree/testing/gtest.h"

namespace {

// hipSuccess is 0 in the HIP ABI.
constexpr int kHipSuccess = 0;

using hipGetProcAddressFn = int (*)(const char* symbol, void** pfn,
                                    int hipVersion, uint64_t flags,
                                    void* symbolStatus);

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

// Opening RTLD_LOCAL is the whole point: it mirrors how Triton loads the HIP
// runtime and is the condition that exposed the bug.
TEST(HipGetProcAddressTest, ResolvesOwnSymbolWhenLoadedRtldLocal) {
  const char* path = CandidateLibPath();
  void* lib = dlopen(path, RTLD_LAZY | RTLD_LOCAL);
  if (lib == nullptr) {
    GTEST_SKIP() << "cannot dlopen " << path << ": " << dlerror();
  }

  auto hip_get_proc_address =
      reinterpret_cast<hipGetProcAddressFn>(dlsym(lib, "hipGetProcAddress"));
  ASSERT_NE(hip_get_proc_address, nullptr)
      << "hipGetProcAddress not exported by " << path;

  void* pfn = nullptr;
  int symbol_status = -1;
  // hipVersion 60000000 == major 6; matches the minimum Triton requires.
  int rc = hip_get_proc_address("hipGetLastError", &pfn, 60000000,
                                /*flags=*/0, &symbol_status);

  EXPECT_EQ(rc, kHipSuccess)
      << "hipGetProcAddress must resolve own symbols under RTLD_LOCAL";
  EXPECT_NE(pfn, nullptr) << "resolved function pointer must be non-null";
  EXPECT_EQ(symbol_status, 0) << "symbolStatus must report success";

  dlclose(lib);
}

}  // namespace
