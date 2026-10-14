// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/symbol_resolver_win32.h"

#include <array>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"

// An exported frame makes symbol resolution testable in stripped test bundles.
// The annotation after allocation keeps the capturing call from being
// tail-called.
extern "C" __declspec(dllexport) __declspec(noinline) iree_status_t
iree_symbol_resolver_test_capture(void) {
  iree_status_t status =
      iree_make_status(IREE_STATUS_UNAVAILABLE, "symbol resolution witness");
  return iree_status_annotate(status, iree_make_cstring_view("captured frame"));
}

namespace {

TEST(SymbolResolverTest, ConcurrentSymbolsAndStatus) {
  constexpr size_t kThreadCount = 8;
  std::array<const iree_symbol_resolver_t*, kThreadCount> resolvers = {};
  std::mutex mutex;
  std::condition_variable condition;
  size_t ready_count = 0;
  bool start = false;
  std::vector<std::thread> threads;
  for (size_t thread_index = 0; thread_index < kThreadCount; ++thread_index) {
    threads.emplace_back([&, thread_index] {
      {
        std::unique_lock<std::mutex> lock(mutex);
        ++ready_count;
        condition.notify_all();
        condition.wait(lock, [&] { return start; });
      }
      const iree_symbol_resolver_t* resolver = nullptr;
      ASSERT_EQ(ERROR_SUCCESS, iree_symbol_resolver_get(&resolver));
      ASSERT_NE(nullptr, resolver);
      resolvers[thread_index] = resolver;
      EXPECT_NE(GetCurrentProcess(), resolver->process);
      EXPECT_EQ(GetCurrentProcessId(), GetProcessId(resolver->process));

      for (int iteration = 0; iteration < 64; ++iteration) {
        alignas(SYMBOL_INFO) unsigned char
            symbol_storage[sizeof(SYMBOL_INFO) + MAX_SYM_NAME] = {};
        auto* symbol = reinterpret_cast<SYMBOL_INFO*>(symbol_storage);
        symbol->SizeOfStruct = sizeof(*symbol);
        symbol->MaxNameLen = MAX_SYM_NAME;
        DWORD64 displacement = 0;
        IMAGEHLP_MODULE64 module = {};
        module.SizeOfStruct = sizeof(module);
        IMAGEHLP_LINE64 line = {};
        line.SizeOfStruct = sizeof(line);
        DWORD line_displacement = 0;
        const DWORD64 address =
            reinterpret_cast<DWORD64>(&iree_symbol_resolver_test_capture);

        iree_symbol_resolver_lock();
        BOOL found_symbol = resolver->SymFromAddr(resolver->process, address,
                                                  &displacement, symbol);
        BOOL found_module =
            resolver->SymGetModuleInfo64(resolver->process, address, &module);
        BOOL found_line = resolver->SymGetLineFromAddr64(
            resolver->process, address, &line_displacement, &line);
        std::string filename = found_line ? line.FileName : "";
        iree_symbol_resolver_unlock();

        ASSERT_TRUE(found_symbol);
        ASSERT_TRUE(found_module);
        EXPECT_STREQ("iree_symbol_resolver_test_capture", symbol->Name);
        EXPECT_EQ(module.BaseOfImage, symbol->ModBase);
        EXPECT_NE('\0', module.ModuleName[0]);
        if (found_line) {
          EXPECT_GT(line.LineNumber, 0u);
          EXPECT_THAT(filename,
                      ::testing::HasSubstr("symbol_resolver_win32_test.cc"));
        }

#if (IREE_STATUS_FEATURES & IREE_STATUS_FEATURE_STACK_TRACE) != 0
        iree_status_t status = iree_symbol_resolver_test_capture();
        char text[16384];
        iree_host_size_t text_length = 0;
        EXPECT_TRUE(
            iree_status_format(status, sizeof(text), text, &text_length));
        EXPECT_THAT(text,
                    ::testing::HasSubstr("<iree_symbol_resolver_test_capture"));
        iree_status_free(status);
#endif  // has IREE_STATUS_FEATURE_STACK_TRACE
      }
    });
  }
  {
    std::unique_lock<std::mutex> lock(mutex);
    condition.wait(lock, [&] { return ready_count == kThreadCount; });
    start = true;
    condition.notify_all();
  }
  for (auto& thread : threads) {
    thread.join();
  }
  for (const auto* resolver : resolvers) {
    EXPECT_EQ(resolvers[0], resolver);
  }
}

}  // namespace
