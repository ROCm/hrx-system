// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <catch2/internal/catch_test_failure_exception.hpp>

#include "hrx_test_fixture.hpp"

TEST_CASE("Runtime version is valid", "[lifecycle]") {
  int major = -1, minor = -1, patch = -1;
  hrx().runtime_version(&major, &minor, &patch);
  REQUIRE(major >= 0);
  REQUIRE(minor >= 0);
  REQUIRE(patch >= 0);
}

TEST_CASE("Selected device belongs to its sealed group", "[lifecycle]") {
  hrx_device_group_t group = hrx().device_get_device_group(g_test_device);
  REQUIRE(group != nullptr);
  REQUIRE(hrx().device_group_device_count(group) == 1);
  REQUIRE(hrx().device_group_get_device(group, 0) == g_test_device);
  REQUIRE(hrx().device_group_get_device(group, 1) == nullptr);
  hrx().device_group_retain(group);
  hrx().device_group_release(group);
  REQUIRE(hrx().device_group_get_device(group, 0) == g_test_device);
}

// This hidden case is launched by cleanup_test.py. The real group is destroyed
// before the dependency wrapper reports a diagnostic outside Catch assertions.
TEST_CASE("Device group cleanup error reaches process exit", "[.cleanup]") {
  REQUIRE(g_test_device_type == HRX_ACCELERATOR_CPU);
  static auto release = hrx().device_group_release;
  hrx().device_group_release = [](hrx_device_group_t group) {
    release(group);
    ReportHrxTestFailure(
        "Device group cleanup",
        hrx().make_status(HRX_STATUS_ABORTED,
                          "device group cleanup error witness"));
  };
}

TEST_CASE("Assertion unwinding releases scoped resources", "[lifecycle]") {
  bool released = false;
  struct ScopedResource {
    ~ScopedResource() { released = true; }
    // Observable owner cleanup, required even when Catch exits an assertion.
    bool& released;
  };
  try {
    ScopedResource resource{released};
    Catch::throw_test_failure_exception();
  } catch (const Catch::TestFailureException&) {
  }
  REQUIRE(released);
}
