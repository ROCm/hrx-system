// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/tls.h"

#include <atomic>
#include <thread>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(StreamingTlsTest, SetGetCurrentThread) {
  iree_hal_streaming_tls_key_t key = IREE_HAL_STREAMING_TLS_KEY_INVALID;
  IREE_ASSERT_OK(iree_hal_streaming_tls_key_create(&key, nullptr));

  int main_value = 1;
  int thread_value = 2;
  IREE_ASSERT_OK(iree_hal_streaming_tls_set(key, &main_value));
  EXPECT_EQ(&main_value, iree_hal_streaming_tls_get(key));

  std::thread thread([&]() {
    EXPECT_EQ(nullptr, iree_hal_streaming_tls_get(key));
    IREE_ASSERT_OK(iree_hal_streaming_tls_set(key, &thread_value));
    EXPECT_EQ(&thread_value, iree_hal_streaming_tls_get(key));
  });
  thread.join();

  EXPECT_EQ(&main_value, iree_hal_streaming_tls_get(key));
  IREE_ASSERT_OK(iree_hal_streaming_tls_set(key, nullptr));
  iree_hal_streaming_tls_key_delete(key);
}

#if !defined(IREE_PLATFORM_WINDOWS)
TEST(StreamingTlsTest, DestructorRunsOnThreadExit) {
  iree_hal_streaming_tls_key_t key = IREE_HAL_STREAMING_TLS_KEY_INVALID;
  IREE_ASSERT_OK(iree_hal_streaming_tls_key_create(
      &key, +[](void* value) {
        auto* destructor_count = static_cast<std::atomic<int>*>(value);
        destructor_count->fetch_add(1, std::memory_order_acq_rel);
      }));

  std::atomic<int> destructor_count{0};
  std::thread thread([&]() {
    IREE_ASSERT_OK(iree_hal_streaming_tls_set(key, &destructor_count));
    EXPECT_EQ(&destructor_count, iree_hal_streaming_tls_get(key));
  });
  thread.join();

  EXPECT_EQ(1, destructor_count.load(std::memory_order_acquire));
  iree_hal_streaming_tls_key_delete(key);
}
#endif  // !IREE_PLATFORM_WINDOWS

TEST(StreamingTlsTest, CapacityExhausted) {
  iree_hal_streaming_tls_key_t keys[IREE_HAL_STREAMING_TLS_KEY_CAPACITY] = {
      IREE_HAL_STREAMING_TLS_KEY_INVALID};
  for (iree_host_size_t i = 0; i < IREE_HAL_STREAMING_TLS_KEY_CAPACITY; ++i) {
    IREE_ASSERT_OK(iree_hal_streaming_tls_key_create(&keys[i], nullptr));
  }

  iree_hal_streaming_tls_key_t extra_key = IREE_HAL_STREAMING_TLS_KEY_INVALID;
  iree_status_t status = iree_hal_streaming_tls_key_create(&extra_key, nullptr);
  EXPECT_EQ(IREE_STATUS_RESOURCE_EXHAUSTED, iree_status_code(status));
  iree_status_ignore(status);
  EXPECT_EQ(IREE_HAL_STREAMING_TLS_KEY_INVALID, extra_key);

  for (iree_host_size_t i = 0; i < IREE_HAL_STREAMING_TLS_KEY_CAPACITY; ++i) {
    iree_hal_streaming_tls_key_delete(keys[i]);
  }
}

}  // namespace
