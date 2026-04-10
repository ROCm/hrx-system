// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef HRX_CTS_TEST_FIXTURE_HPP
#define HRX_CTS_TEST_FIXTURE_HPP

#include "hrx_loader.hpp"

// Global test device (set by main.cpp).
extern hrx_device_t g_test_device;
extern hrx_accelerator_type_t g_test_device_type;

class HrxTestFixture {
 protected:
  hrx_device_t device_ = nullptr;

  HrxTestFixture() { device_ = g_test_device; }

  ~HrxTestFixture() {
    if (device_) {
      hrx().device_synchronize(device_);
    }
  }

  bool is_gpu() const { return g_test_device_type == HRX_ACCELERATOR_GPU; }
  bool is_cpu() const { return g_test_device_type == HRX_ACCELERATOR_CPU; }
};

// Convenience: check hrx_status_t in tests.
#define REQUIRE_OK(expr)                                      \
  do {                                                        \
    hrx_status_t _s = (expr);                                \
    if (!hrx_status_is_ok(_s)) {                             \
      char* msg = nullptr;                                    \
      size_t len = 0;                                         \
      hrx().status_to_string(_s, &msg, &len);                \
      INFO("hrx error: " << (msg ? msg : "?"));              \
      hrx().status_free_message(msg);                        \
      hrx().status_ignore(_s);                               \
      REQUIRE(false);                                         \
    }                                                         \
  } while (0)

#endif  // HRX_CTS_TEST_FIXTURE_HPP
