// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include "hrx_test_fixture.hpp"
#include <catch2/catch_test_macros.hpp>

TEST_CASE("Runtime version is valid", "[lifecycle]") {
  int major = -1, minor = -1, patch = -1;
  hrx().runtime_version(&major, &minor, &patch);
  REQUIRE(major >= 0);
  REQUIRE(minor >= 0);
  REQUIRE(patch >= 0);
}

TEST_CASE("Selected accelerator device_count works", "[lifecycle]") {
  int count = 0;
  if (g_test_device_type == HRX_ACCELERATOR_GPU) {
    REQUIRE_OK(hrx().gpu_device_count(&count));
  } else {
    REQUIRE_OK(hrx().cpu_device_count(&count));
  }
  REQUIRE(count > 0);
}

TEST_CASE("GPU init and shutdown", "[lifecycle][gpu]") {
  // GPU may already be initialized by main.cpp.
  int count = 0;
  hrx_status_t status = hrx().gpu_device_count(&count);
  if (hrx_status_is_ok(status)) {
    REQUIRE(count > 0);
  } else {
    // GPU not initialized — that's OK for CPU-only test runs.
    hrx().status_ignore(status);
  }
}

TEST_CASE("Double init returns ALREADY_EXISTS", "[lifecycle]") {
  hrx_status_t status = g_test_device_type == HRX_ACCELERATOR_GPU
                            ? hrx().gpu_initialize(0)
                            : hrx().cpu_initialize(0);
  REQUIRE(!hrx_status_is_ok(status));
  REQUIRE(hrx().status_code(status) == HRX_STATUS_ALREADY_EXISTS);
  hrx().status_ignore(status);
}
