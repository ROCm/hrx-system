// Copyright 2026 The Pyre Authors
// SPDX-License-Identifier: Apache-2.0

#include "pyre_test_fixture.hpp"
#include <catch2/catch_test_macros.hpp>

TEST_CASE("Runtime version is valid", "[lifecycle]") {
  int major = -1, minor = -1, patch = -1;
  pyre().runtime_version(&major, &minor, &patch);
  REQUIRE(major >= 0);
  REQUIRE(minor >= 0);
  REQUIRE(patch >= 0);
}

TEST_CASE("CPU init and shutdown", "[lifecycle][cpu]") {
  // CPU is already initialized by main.cpp if we're on CPU.
  // Test that device_count works.
  int count = 0;
  REQUIRE_OK(pyre().cpu_device_count(&count));
  REQUIRE(count > 0);
}

TEST_CASE("GPU init and shutdown", "[lifecycle][gpu]") {
  // GPU may already be initialized by main.cpp.
  int count = 0;
  pyre_status_t status = pyre().gpu_device_count(&count);
  if (pyre_status_is_ok(status)) {
    REQUIRE(count > 0);
  } else {
    // GPU not initialized — that's OK for CPU-only test runs.
    pyre().status_ignore(status);
  }
}

TEST_CASE("Double init returns ALREADY_EXISTS", "[lifecycle]") {
  // CPU is already initialized. Calling again should return error.
  pyre_status_t status = pyre().cpu_initialize(0);
  REQUIRE(!pyre_status_is_ok(status));
  REQUIRE(pyre().status_code(status) == PYRE_STATUS_ALREADY_EXISTS);
  pyre().status_ignore(status);
}
