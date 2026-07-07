// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

// Regression test for hipGetProcAddress()-family symbol resolution scope.
//
// hipGetProcAddress() must resolve symbols against the HIP runtime library
// that defines them, NOT the process-global symbol scope. Consumers such as
// Triton's AMD backend dlopen() libamdhip64 with RTLD_LOCAL and then resolve
// the whole HIP API through hipGetProcAddress(). When these entry points were
// implemented as dlsym(dlopen(NULL), symbol) they searched only the global
// scope, so an RTLD_LOCAL load left our symbols invisible and every lookup
// (starting with the very first, hipGetLastError) failed with
// hipErrorNotFound -- taking down Triton at driver-init time. See issue for
// details.
//
// The same lookup backs the stream-per-thread variants
// (hipGetProcAddress_spt / hipGetDriverEntryPoint_spt via hrx_hip_spt_lookup),
// so those had the identical bug and are exercised here too.
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

// Resolves `entry_point` from the RTLD_LOCAL-loaded library and asserts it can
// look up a core HIP symbol (hipGetLastError, the first thing Triton needs).
// Opening RTLD_LOCAL is the whole point: it mirrors how Triton loads the HIP
// runtime and is the condition that exposed the bug.
void ExpectResolvesUnderRtldLocal(const char* entry_point) {
  const char* path = CandidateLibPath();
  void* lib = dlopen(path, RTLD_LAZY | RTLD_LOCAL);
  if (lib == nullptr) {
    GTEST_SKIP() << "cannot dlopen " << path << ": " << dlerror();
  }

  auto get_proc_address =
      reinterpret_cast<hipGetProcAddressFn>(dlsym(lib, entry_point));
  ASSERT_NE(get_proc_address, nullptr)
      << entry_point << " not exported by " << path;

  void* pfn = nullptr;
  int symbol_status = -1;
  // hipVersion 60000000 == major 6; matches the minimum Triton requires.
  int rc = get_proc_address("hipGetLastError", &pfn, 60000000,
                            /*flags=*/0, &symbol_status);

  EXPECT_EQ(rc, kHipSuccess)
      << entry_point << " must resolve own symbols under RTLD_LOCAL";
  EXPECT_NE(pfn, nullptr) << "resolved function pointer must be non-null";
  EXPECT_EQ(symbol_status, 0) << "symbolStatus must report success";

  dlclose(lib);
}

TEST(HipGetProcAddressTest, ResolvesOwnSymbolWhenLoadedRtldLocal) {
  ExpectResolvesUnderRtldLocal("hipGetProcAddress");
}

// The stream-per-thread entry point shares the same underlying lookup
// (hrx_hip_spt_lookup) and had the identical process-global-scope bug.
TEST(HipGetProcAddressTest, SptEntryPointResolvesOwnSymbolWhenLoadedRtldLocal) {
  ExpectResolvesUnderRtldLocal("hipGetProcAddress_spt");
}

}  // namespace
