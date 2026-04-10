// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include "hrx_test_fixture.hpp"
#include <catch2/catch_test_macros.hpp>

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
