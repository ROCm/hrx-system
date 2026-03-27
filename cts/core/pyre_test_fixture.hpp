// Copyright 2026 The Pyre Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef PYRE_CTS_TEST_FIXTURE_HPP
#define PYRE_CTS_TEST_FIXTURE_HPP

#include "pyre_loader.hpp"

// Global test device (set by main.cpp).
extern pyre_device_t g_test_device;
extern pyre_accelerator_type_t g_test_device_type;

class PyreTestFixture {
 protected:
  pyre_device_t device_ = nullptr;

  PyreTestFixture() { device_ = g_test_device; }

  ~PyreTestFixture() {
    if (device_) {
      pyre().device_synchronize(device_);
    }
  }

  bool is_gpu() const { return g_test_device_type == PYRE_ACCELERATOR_GPU; }
  bool is_cpu() const { return g_test_device_type == PYRE_ACCELERATOR_CPU; }
};

// Convenience: check pyre_status_t in tests.
#define REQUIRE_OK(expr)                                      \
  do {                                                        \
    pyre_status_t _s = (expr);                                \
    if (!pyre_status_is_ok(_s)) {                             \
      char* msg = nullptr;                                    \
      size_t len = 0;                                         \
      pyre().status_to_string(_s, &msg, &len);                \
      INFO("pyre error: " << (msg ? msg : "?"));              \
      pyre().status_free_message(msg);                        \
      pyre().status_ignore(_s);                               \
      REQUIRE(false);                                         \
    }                                                         \
  } while (0)

#endif  // PYRE_CTS_TEST_FIXTURE_HPP
