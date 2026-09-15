// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "binding/hip/function_handle.h"

#include <atomic>
#include <thread>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class FunctionHandleTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_atomic_ref_count_init(&module_.ref_count);
    module_.host_allocator = iree_allocator_system();
    symbol_.module = &module_;
    symbol_.type = IREE_HAL_STREAMING_SYMBOL_TYPE_FUNCTION;
  }

  iree_hal_streaming_module_t module_ = {};
  iree_hal_streaming_symbol_t symbol_ = {};
};

TEST_F(FunctionHandleTest, RejectsStaleAndFabricatedHandles) {
  void* first_handle = nullptr;
  IREE_ASSERT_OK(
      iree_hip_function_handle_create(&module_, &symbol_, &first_handle));

  iree_hal_streaming_symbol_t* resolved_symbol = nullptr;
  iree_hal_streaming_module_t* resolved_module = nullptr;
  EXPECT_TRUE(iree_hip_function_handle_lookup(first_handle, &resolved_symbol,
                                              &resolved_module));
  EXPECT_EQ(resolved_symbol, &symbol_);
  EXPECT_EQ(resolved_module, &module_);
  iree_hal_streaming_module_release(resolved_module);

  iree_hip_function_handle_retire_module(&module_);
  EXPECT_FALSE(iree_hip_function_handle_lookup(first_handle, &resolved_symbol,
                                               &resolved_module));
  EXPECT_FALSE(iree_hip_function_handle_lookup(
      reinterpret_cast<void*>(IREE_HAL_STREAMING_SYMBOL_TAG_VALUE | 0xFFFF),
      &resolved_symbol, &resolved_module));

  void* second_handle = nullptr;
  IREE_ASSERT_OK(
      iree_hip_function_handle_create(&module_, &symbol_, &second_handle));
  EXPECT_NE(first_handle, second_handle);
  iree_hip_function_handle_retire_module(&module_);
}

TEST_F(FunctionHandleTest, ConcurrentLookupAndRetirementPreserveOwner) {
  void* handle = nullptr;
  IREE_ASSERT_OK(iree_hip_function_handle_create(&module_, &symbol_, &handle));

  constexpr int kThreadCount = 8;
  std::atomic<bool> start{false};
  std::atomic<bool> stop{false};
  std::atomic<uint64_t> successful_lookups{0};
  std::vector<std::thread> threads;
  threads.reserve(kThreadCount);
  for (int i = 0; i < kThreadCount; ++i) {
    threads.emplace_back([&] {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      while (!stop.load(std::memory_order_acquire)) {
        iree_hal_streaming_symbol_t* resolved_symbol = nullptr;
        iree_hal_streaming_module_t* resolved_module = nullptr;
        if (iree_hip_function_handle_lookup(handle, &resolved_symbol,
                                            &resolved_module)) {
          EXPECT_EQ(resolved_symbol, &symbol_);
          EXPECT_EQ(resolved_module, &module_);
          successful_lookups.fetch_add(1, std::memory_order_relaxed);
          iree_hal_streaming_module_release(resolved_module);
        }
      }
    });
  }

  start.store(true, std::memory_order_release);
  while (successful_lookups.load(std::memory_order_acquire) == 0) {
    std::this_thread::yield();
  }
  iree_hip_function_handle_retire_module(&module_);
  stop.store(true, std::memory_order_release);
  for (std::thread& thread : threads) thread.join();

  iree_hal_streaming_symbol_t* resolved_symbol = nullptr;
  iree_hal_streaming_module_t* resolved_module = nullptr;
  EXPECT_FALSE(iree_hip_function_handle_lookup(handle, &resolved_symbol,
                                               &resolved_module));
}

}  // namespace
