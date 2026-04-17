// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include "hrx_test_fixture.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cstring>
#include <vector>

namespace {

hrx_status_t hostCallIncrement(void *user_data) {
  auto *value = static_cast<int *>(user_data);
  *value += 1;
  return hrx_ok_status();
}

} // namespace

TEST_CASE_METHOD(HrxTestFixture, "Queue host call", "[queue_ops][host_call]") {
  hrx_semaphore_t done = nullptr;
  REQUIRE_OK(hrx().semaphore_create(device_, 0, &done));

  hrx_semaphore_t signal_semaphores[] = {done};
  uint64_t signal_values[] = {1};
  hrx_semaphore_list_t signal_list = {
      signal_semaphores,
      signal_values,
      1,
  };

  int callback_count = 0;
  REQUIRE_OK(hrx().queue_host_call(device_, 0, nullptr, &signal_list,
                                   hostCallIncrement, &callback_count));
  REQUIRE_OK(hrx().semaphore_wait(done, 1, UINT64_MAX));
  REQUIRE(callback_count == 1);

  hrx().semaphore_release(done);
}

TEST_CASE_METHOD(HrxTestFixture, "Queue fill signals completion semaphore",
                 "[queue_ops][fill][sync]") {
  hrx_allocator_t alloc = hrx().device_allocator(device_);

  hrx_buffer_params_t params = {};
  params.type = HRX_MEMORY_TYPE_DEVICE_LOCAL;
  params.access = HRX_MEMORY_ACCESS_ALL;
  params.usage = HRX_BUFFER_USAGE_TRANSFER;

  constexpr size_t kWordCount = 4096;
  constexpr uint32_t kPattern = 0xA5C33CA5u;
  hrx_buffer_t buf = nullptr;
  REQUIRE_OK(hrx().allocator_allocate_buffer(
      alloc, params, kWordCount * sizeof(uint32_t), &buf));

  hrx_semaphore_t done = nullptr;
  REQUIRE_OK(hrx().semaphore_create(device_, 0, &done));
  hrx_semaphore_t signal_semaphores[] = {done};
  uint64_t signal_values[] = {1};
  hrx_semaphore_list_t signal_list = {
      signal_semaphores,
      signal_values,
      1,
  };

  REQUIRE_OK(hrx().queue_fill(device_, 0, nullptr, &signal_list, buf, 0,
                              kWordCount * sizeof(uint32_t), &kPattern,
                              sizeof(kPattern)));
  REQUIRE_OK(hrx().semaphore_wait(done, 1, UINT64_MAX));

  std::vector<uint32_t> readback(kWordCount);
  REQUIRE_OK(hrx().synchronous_d2h(device_, buf, 0, readback.data(),
                                   readback.size() * sizeof(uint32_t)));
  for (uint32_t value : readback) {
    REQUIRE(value == kPattern);
  }

  hrx().semaphore_release(done);
  hrx().buffer_release(buf);
}

TEST_CASE_METHOD(HrxTestFixture, "Queue copy signals completion semaphore",
                 "[queue_ops][copy][sync]") {
  hrx_allocator_t alloc = hrx().device_allocator(device_);

  hrx_buffer_params_t params = {};
  params.type = HRX_MEMORY_TYPE_DEVICE_LOCAL;
  params.access = HRX_MEMORY_ACCESS_ALL;
  params.usage = HRX_BUFFER_USAGE_TRANSFER;

  constexpr size_t kByteCount = 16 * 1024;
  hrx_buffer_t src = nullptr;
  hrx_buffer_t dst = nullptr;
  REQUIRE_OK(hrx().allocator_allocate_buffer(alloc, params, kByteCount, &src));
  REQUIRE_OK(hrx().allocator_allocate_buffer(alloc, params, kByteCount, &dst));

  std::vector<uint8_t> expected(kByteCount);
  for (size_t i = 0; i < expected.size(); ++i) {
    expected[i] = static_cast<uint8_t>((i * 13u + 7u) & 0xFFu);
  }
  REQUIRE_OK(
      hrx().synchronous_h2d(device_, expected.data(), src, 0, expected.size()));

  hrx_semaphore_t done = nullptr;
  REQUIRE_OK(hrx().semaphore_create(device_, 0, &done));
  hrx_semaphore_t signal_semaphores[] = {done};
  uint64_t signal_values[] = {1};
  hrx_semaphore_list_t signal_list = {
      signal_semaphores,
      signal_values,
      1,
  };

  REQUIRE_OK(hrx().queue_copy(device_, 0, nullptr, &signal_list, src, 0, dst, 0,
                              kByteCount));
  REQUIRE_OK(hrx().semaphore_wait(done, 1, UINT64_MAX));

  std::vector<uint8_t> readback(kByteCount);
  REQUIRE_OK(
      hrx().synchronous_d2h(device_, dst, 0, readback.data(), readback.size()));
  REQUIRE(std::memcmp(readback.data(), expected.data(), expected.size()) == 0);

  hrx().semaphore_release(done);
  hrx().buffer_release(dst);
  hrx().buffer_release(src);
}
