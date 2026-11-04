// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef HRX_CTS_TEST_FIXTURE_HPP
#define HRX_CTS_TEST_FIXTURE_HPP

#include <string>

#include "hrx_loader.hpp"

// Global test device (set by main.cpp).
extern hrx_device_t g_test_device;
extern hrx_accelerator_type_t g_test_device_type;
extern std::string g_test_hip_library_path;

// Reports and consumes a failed status, marking the process unsuccessful even
// when no Catch assertion can run. Safe during fixture destruction.
void ReportHrxTestFailure(const char* context, hrx_status_t status) noexcept;

class HrxTestFixture {
 protected:
  // Borrowed device owned by the CTS process until accelerator shutdown.
  hrx_device_t device_ = g_test_device;

  bool is_gpu() const { return g_test_device_type == HRX_ACCELERATOR_GPU; }
  bool is_cpu() const { return g_test_device_type == HRX_ACCELERATOR_CPU; }
};

// Convenience: check hrx_status_t in tests.
#define REQUIRE_OK(expr)               \
  do {                                 \
    hrx_status_t _s = (expr);          \
    if (!hrx_status_is_ok(_s)) {       \
      ReportHrxTestFailure(#expr, _s); \
      REQUIRE(false);                  \
    }                                  \
  } while (0)

#endif  // HRX_CTS_TEST_FIXTURE_HPP
