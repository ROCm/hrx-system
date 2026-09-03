// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

// Regression tests for dynamic HIP entry-point resolution.
//
// These entry points must resolve against the HIP runtime library that defines
// them. A process-global lookup cannot find a HIP runtime opened with
// RTLD_LOCAL, even though the caller has a valid entry-point function pointer
// from that library.
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
#include <limits>

#include "iree/testing/gtest.h"

namespace {

// hipSuccess is 0 in the HIP ABI.
constexpr int kHipSuccess = 0;
constexpr int kHipErrorInvalidValue = 1;
constexpr uint64_t kHipGetProcAddressDefault = 0;
constexpr uint64_t kHipGetProcAddressLegacyStream = 1;
constexpr uint64_t kHipGetProcAddressPerThreadDefaultStream = 2;
constexpr uint64_t kHipGetProcAddressInvalidFlags =
    std::numeric_limits<uint64_t>::max();
constexpr int kHipGetProcAddressSymbolNotFound = 1;
constexpr int kHipVersionBeforeDevicePropertiesR0600 = 599;
constexpr int kHipVersionWithDevicePropertiesR0600 = 600;

using hipGetProcAddressFn = int (*)(const char* symbol, void** pfn,
                                    int hipVersion, uint64_t flags,
                                    void* symbolStatus);
using hipGetDriverEntryPointFn = int (*)(const char* symbol, void** pfn,
                                         unsigned long long flags,
                                         void* symbolStatus);
using hipGetLastErrorFn = int (*)();

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

// Resolves an entry point from an RTLD_LOCAL-loaded runtime and verifies that
// it can look up a core HIP symbol.
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
  // hipVersion 60000000 selects the version-6 entry-point aliases.
  int rc = get_proc_address("hipGetLastError", &pfn, 60000000,
                            kHipGetProcAddressDefault, &symbol_status);

  EXPECT_EQ(rc, kHipSuccess)
      << entry_point << " must resolve own symbols under RTLD_LOCAL";
  EXPECT_NE(pfn, nullptr) << "resolved function pointer must be non-null";
  EXPECT_EQ(symbol_status, 0) << "symbolStatus must report success";

  dlclose(lib);
}

void ExpectRejectsInvalidFlagsUnderRtldLocal(const char* entry_point) {
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
  int rc = get_proc_address("hipGetLastError", &pfn, 60000000,
                            kHipGetProcAddressInvalidFlags, &symbol_status);

  EXPECT_EQ(rc, kHipErrorInvalidValue)
      << entry_point << " must reject an unsupported lookup mode";

  dlclose(lib);
}

void ExpectDriverEntryPointResolvesUnderRtldLocal(const char* entry_point) {
  const char* path = CandidateLibPath();
  void* lib = dlopen(path, RTLD_LAZY | RTLD_LOCAL);
  if (lib == nullptr) {
    GTEST_SKIP() << "cannot dlopen " << path << ": " << dlerror();
  }

  auto get_driver_entry_point =
      reinterpret_cast<hipGetDriverEntryPointFn>(dlsym(lib, entry_point));
  ASSERT_NE(get_driver_entry_point, nullptr)
      << entry_point << " not exported by " << path;

  void* pfn = nullptr;
  int symbol_status = -1;
  int rc = get_driver_entry_point("hipGetLastError", &pfn,
                                  kHipGetProcAddressDefault, &symbol_status);

  EXPECT_EQ(rc, kHipSuccess)
      << entry_point << " must resolve own symbols under RTLD_LOCAL";
  EXPECT_NE(pfn, nullptr) << "resolved function pointer must be non-null";
  EXPECT_EQ(symbol_status, 0) << "symbolStatus must report success";

  dlclose(lib);
}

void ExpectDriverEntryPointRejectsInvalidFlagsUnderRtldLocal(
    const char* entry_point) {
  const char* path = CandidateLibPath();
  void* lib = dlopen(path, RTLD_LAZY | RTLD_LOCAL);
  if (lib == nullptr) {
    GTEST_SKIP() << "cannot dlopen " << path << ": " << dlerror();
  }

  auto get_driver_entry_point =
      reinterpret_cast<hipGetDriverEntryPointFn>(dlsym(lib, entry_point));
  ASSERT_NE(get_driver_entry_point, nullptr)
      << entry_point << " not exported by " << path;

  void* pfn = nullptr;
  int symbol_status = -1;
  int rc = get_driver_entry_point(
      "hipGetLastError", &pfn, kHipGetProcAddressInvalidFlags, &symbol_status);

  EXPECT_EQ(rc, kHipErrorInvalidValue)
      << entry_point << " must reject an unsupported lookup mode";

  dlclose(lib);
}

void ExpectLookupResult(hipGetProcAddressFn get_proc_address,
                        const char* symbol, int hip_version, uint64_t flags,
                        void* expected) {
  void* pfn = nullptr;
  int symbol_status = -1;
  int rc = get_proc_address(symbol, &pfn, hip_version, flags, &symbol_status);
  EXPECT_EQ(rc, kHipSuccess);
  EXPECT_EQ(pfn, expected);
  EXPECT_EQ(symbol_status, 0);
}

void ExpectDriverLookupResult(hipGetDriverEntryPointFn get_driver_entry_point,
                              const char* symbol, uint64_t flags,
                              void* expected) {
  void* pfn = nullptr;
  int symbol_status = -1;
  int rc = get_driver_entry_point(symbol, &pfn, flags, &symbol_status);
  EXPECT_EQ(rc, kHipSuccess);
  EXPECT_EQ(pfn, expected);
  EXPECT_EQ(symbol_status, 0);
}

TEST(HipGetProcAddressTest, ResolvesOwnSymbolWhenLoadedRtldLocal) {
  ExpectResolvesUnderRtldLocal("hipGetProcAddress");
}

TEST(HipGetProcAddressTest, SptEntryPointResolvesOwnSymbolWhenLoadedRtldLocal) {
  ExpectResolvesUnderRtldLocal("hipGetProcAddress_spt");
}

TEST(HipGetProcAddressTest, RejectsInvalidLookupFlagsWhenLoadedRtldLocal) {
  ExpectRejectsInvalidFlagsUnderRtldLocal("hipGetProcAddress");
  ExpectRejectsInvalidFlagsUnderRtldLocal("hipGetProcAddress_spt");
}

TEST(HipGetProcAddressTest,
     DriverEntryPointsResolveOwnSymbolsWhenLoadedRtldLocal) {
  ExpectDriverEntryPointResolvesUnderRtldLocal("hipGetDriverEntryPoint");
  ExpectDriverEntryPointResolvesUnderRtldLocal("hipGetDriverEntryPoint_spt");
}

TEST(HipGetProcAddressTest,
     DriverEntryPointsRejectInvalidLookupFlagsWhenLoadedRtldLocal) {
  ExpectDriverEntryPointRejectsInvalidFlagsUnderRtldLocal(
      "hipGetDriverEntryPoint");
  ExpectDriverEntryPointRejectsInvalidFlagsUnderRtldLocal(
      "hipGetDriverEntryPoint_spt");
}

TEST(HipGetProcAddressTest, SelectsRequestedDefaultStreamConvention) {
  const char* path = CandidateLibPath();
  void* lib = dlopen(path, RTLD_LAZY | RTLD_LOCAL);
  if (lib == nullptr) {
    GTEST_SKIP() << "cannot dlopen " << path << ": " << dlerror();
  }

  auto get_proc_address =
      reinterpret_cast<hipGetProcAddressFn>(dlsym(lib, "hipGetProcAddress"));
  auto get_proc_address_spt = reinterpret_cast<hipGetProcAddressFn>(
      dlsym(lib, "hipGetProcAddress_spt"));
  auto get_driver_entry_point = reinterpret_cast<hipGetDriverEntryPointFn>(
      dlsym(lib, "hipGetDriverEntryPoint"));
  auto get_driver_entry_point_spt = reinterpret_cast<hipGetDriverEntryPointFn>(
      dlsym(lib, "hipGetDriverEntryPoint_spt"));
  ASSERT_NE(get_proc_address, nullptr);
  ASSERT_NE(get_proc_address_spt, nullptr);
  ASSERT_NE(get_driver_entry_point, nullptr);
  ASSERT_NE(get_driver_entry_point_spt, nullptr);

  void* legacy_symbol = dlsym(lib, "hipMemcpyAsync");
  void* per_thread_symbol = dlsym(lib, "hipMemcpyAsync_spt");
  ASSERT_NE(legacy_symbol, nullptr);
  ASSERT_NE(per_thread_symbol, nullptr);
  ASSERT_NE(legacy_symbol, per_thread_symbol);

  ExpectLookupResult(get_proc_address, "hipMemcpyAsync", 600,
                     kHipGetProcAddressDefault, legacy_symbol);
  ExpectLookupResult(get_proc_address, "hipMemcpyAsync", 600,
                     kHipGetProcAddressPerThreadDefaultStream,
                     per_thread_symbol);
  ExpectLookupResult(get_proc_address_spt, "hipMemcpyAsync", 600,
                     kHipGetProcAddressDefault, per_thread_symbol);
  ExpectLookupResult(get_proc_address_spt, "hipMemcpyAsync", 600,
                     kHipGetProcAddressLegacyStream, legacy_symbol);

  ExpectDriverLookupResult(get_driver_entry_point, "hipMemcpyAsync",
                           kHipGetProcAddressDefault, legacy_symbol);
  ExpectDriverLookupResult(get_driver_entry_point, "hipMemcpyAsync",
                           kHipGetProcAddressPerThreadDefaultStream,
                           per_thread_symbol);
  ExpectDriverLookupResult(get_driver_entry_point_spt, "hipMemcpyAsync",
                           kHipGetProcAddressDefault, per_thread_symbol);
  ExpectDriverLookupResult(get_driver_entry_point_spt, "hipMemcpyAsync",
                           kHipGetProcAddressLegacyStream, legacy_symbol);

  dlclose(lib);
}

TEST(HipGetProcAddressTest, SelectsVersionedDevicePropertyAbi) {
  const char* path = CandidateLibPath();
  void* lib = dlopen(path, RTLD_LAZY | RTLD_LOCAL);
  if (lib == nullptr) {
    GTEST_SKIP() << "cannot dlopen " << path << ": " << dlerror();
  }

  auto get_proc_address =
      reinterpret_cast<hipGetProcAddressFn>(dlsym(lib, "hipGetProcAddress"));
  ASSERT_NE(get_proc_address, nullptr);
  void* properties_r0000 = dlsym(lib, "hipGetDevicePropertiesR0000");
  void* properties_r0600 = dlsym(lib, "hipGetDevicePropertiesR0600");
  ASSERT_NE(properties_r0000, nullptr);
  ASSERT_NE(properties_r0600, nullptr);

  ExpectLookupResult(get_proc_address, "hipGetDeviceProperties",
                     kHipVersionBeforeDevicePropertiesR0600,
                     kHipGetProcAddressPerThreadDefaultStream,
                     properties_r0000);
  ExpectLookupResult(get_proc_address, "hipGetDeviceProperties",
                     kHipVersionWithDevicePropertiesR0600,
                     kHipGetProcAddressPerThreadDefaultStream,
                     properties_r0600);

  dlclose(lib);
}

TEST(HipGetProcAddressTest, ReportsMissingSymbolsAndPublishesLastError) {
  const char* path = CandidateLibPath();
  void* lib = dlopen(path, RTLD_LAZY | RTLD_LOCAL);
  if (lib == nullptr) {
    GTEST_SKIP() << "cannot dlopen " << path << ": " << dlerror();
  }

  auto get_proc_address =
      reinterpret_cast<hipGetProcAddressFn>(dlsym(lib, "hipGetProcAddress"));
  auto get_last_error =
      reinterpret_cast<hipGetLastErrorFn>(dlsym(lib, "hipGetLastError"));
  ASSERT_NE(get_proc_address, nullptr);
  ASSERT_NE(get_last_error, nullptr);

  void* pfn = reinterpret_cast<void*>(uintptr_t{1});
  int symbol_status = -1;
  int rc = get_proc_address("hipSymbolThatDoesNotExist", &pfn, 600,
                            kHipGetProcAddressDefault, &symbol_status);
  EXPECT_EQ(rc, kHipErrorInvalidValue);
  EXPECT_EQ(pfn, nullptr);
  EXPECT_EQ(symbol_status, kHipGetProcAddressSymbolNotFound);
  EXPECT_EQ(get_last_error(), kHipErrorInvalidValue);

  dlclose(lib);
}

}  // namespace
