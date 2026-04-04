// Copyright 2026 The Pyre Authors
// SPDX-License-Identifier: Apache-2.0

#include "pyre_test_fixture.hpp"
#include <catch2/catch_test_macros.hpp>

namespace {

pyre_status_t hostCallIncrement(void* user_data) {
  auto* value = static_cast<int*>(user_data);
  *value += 1;
  return pyre_ok_status();
}

} // namespace

TEST_CASE_METHOD(PyreTestFixture, "Queue host call",
                 "[queue_ops][host_call]") {
  pyre_semaphore_t done = nullptr;
  REQUIRE_OK(pyre().semaphore_create(device_, 0, &done));

  pyre_semaphore_t signal_semaphores[] = {done};
  uint64_t signal_values[] = {1};
  pyre_semaphore_list_t signal_list = {
      signal_semaphores,
      signal_values,
      1,
  };

  int callback_count = 0;
  REQUIRE_OK(pyre().queue_host_call(
      device_, 0, nullptr, &signal_list, hostCallIncrement,
      &callback_count));
  REQUIRE_OK(pyre().semaphore_wait(done, 1, UINT64_MAX));
  REQUIRE(callback_count == 1);

  pyre().semaphore_release(done);
}
