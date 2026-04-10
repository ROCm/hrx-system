// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include "hrx_test_fixture.hpp"
#include <catch2/catch_test_macros.hpp>

TEST_CASE_METHOD(HrxTestFixture, "Semaphore create and query", "[semaphore]") {
  hrx_semaphore_t sem = nullptr;
  REQUIRE_OK(hrx().semaphore_create(device_, 0, &sem));

  uint64_t value = UINT64_MAX;
  REQUIRE_OK(hrx().semaphore_query(sem, &value));
  REQUIRE(value == 0);

  hrx().semaphore_release(sem);
}

TEST_CASE_METHOD(HrxTestFixture, "Semaphore signal and query",
                 "[semaphore][sync]") {
  hrx_semaphore_t sem = nullptr;
  REQUIRE_OK(hrx().semaphore_create(device_, 0, &sem));

  REQUIRE_OK(hrx().semaphore_signal(sem, 42));

  uint64_t value = 0;
  REQUIRE_OK(hrx().semaphore_query(sem, &value));
  REQUIRE(value == 42);

  hrx().semaphore_release(sem);
}

TEST_CASE_METHOD(HrxTestFixture, "Semaphore wait after signal",
                 "[semaphore][sync]") {
  hrx_semaphore_t sem = nullptr;
  REQUIRE_OK(hrx().semaphore_create(device_, 0, &sem));

  REQUIRE_OK(hrx().semaphore_signal(sem, 1));
  REQUIRE_OK(hrx().semaphore_wait(sem, 1, UINT64_MAX));

  hrx().semaphore_release(sem);
}

TEST_CASE_METHOD(HrxTestFixture, "Semaphore poll returns immediately",
                 "[semaphore][sync]") {
  hrx_semaphore_t sem = nullptr;
  REQUIRE_OK(hrx().semaphore_create(device_, 5, &sem));

  // Wait with timeout=0 (poll) for already-reached value.
  REQUIRE_OK(hrx().semaphore_wait(sem, 5, 0));

  hrx().semaphore_release(sem);
}
