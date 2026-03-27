// Copyright 2026 The Pyre Authors
// SPDX-License-Identifier: Apache-2.0

#include "pyre_test_fixture.hpp"
#include <catch2/catch_test_macros.hpp>

TEST_CASE_METHOD(PyreTestFixture, "Semaphore create and query",
                 "[semaphore]") {
  pyre_semaphore_t sem = nullptr;
  REQUIRE_OK(pyre().semaphore_create(device_, 0, &sem));

  uint64_t value = UINT64_MAX;
  REQUIRE_OK(pyre().semaphore_query(sem, &value));
  REQUIRE(value == 0);

  REQUIRE_OK(pyre().semaphore_release(sem));
}

TEST_CASE_METHOD(PyreTestFixture, "Semaphore signal and query",
                 "[semaphore][sync]") {
  pyre_semaphore_t sem = nullptr;
  REQUIRE_OK(pyre().semaphore_create(device_, 0, &sem));

  REQUIRE_OK(pyre().semaphore_signal(sem, 42));

  uint64_t value = 0;
  REQUIRE_OK(pyre().semaphore_query(sem, &value));
  REQUIRE(value == 42);

  REQUIRE_OK(pyre().semaphore_release(sem));
}

TEST_CASE_METHOD(PyreTestFixture, "Semaphore wait after signal",
                 "[semaphore][sync]") {
  pyre_semaphore_t sem = nullptr;
  REQUIRE_OK(pyre().semaphore_create(device_, 0, &sem));

  REQUIRE_OK(pyre().semaphore_signal(sem, 1));
  REQUIRE_OK(pyre().semaphore_wait(sem, 1, UINT64_MAX));

  REQUIRE_OK(pyre().semaphore_release(sem));
}

TEST_CASE_METHOD(PyreTestFixture, "Semaphore poll returns immediately",
                 "[semaphore][sync]") {
  pyre_semaphore_t sem = nullptr;
  REQUIRE_OK(pyre().semaphore_create(device_, 5, &sem));

  // Wait with timeout=0 (poll) for already-reached value.
  REQUIRE_OK(pyre().semaphore_wait(sem, 5, 0));

  REQUIRE_OK(pyre().semaphore_release(sem));
}
