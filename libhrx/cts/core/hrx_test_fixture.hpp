// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef HRX_CTS_TEST_FIXTURE_HPP
#define HRX_CTS_TEST_FIXTURE_HPP

#include <catch2/catch_test_macros.hpp>
#include <string>

#include "hrx_loader.hpp"

// Global test device (set by main.cpp).
extern hrx_device_t g_test_device;
extern hrx_accelerator_type_t g_test_device_type;
extern std::string g_test_hip_library_path;

// Reports and consumes a failed status, marking the process unsuccessful even
// when no Catch assertion can run. Safe during fixture destruction.
void ReportHrxTestFailure(const char* context, hrx_status_t status) noexcept;

// Runtime event sink for both harness-owned and explicitly constructed groups.
// Error events remain process failures even during final device destruction.
void ReportHrxDeviceEvent(void* user_data,
                          const hrx_device_event_t* event) noexcept;

// Selects a provisioned queue with the operations required by a CTS scenario.
// Selection belongs to the caller; the runtime never substitutes another queue.
inline hrx_queue_t FindProvisionedQueue(hrx_device_t device,
                                        hrx_queue_role_flags_t roles) {
  for (uint32_t i = 0; i < hrx().device_queue_family_count(device); ++i) {
    hrx_queue_family_info_t info;
    hrx_status_t status = hrx().device_query_queue_family(device, i, &info);
    if (!hrx_status_is_ok(status)) {
      ReportHrxTestFailure("query queue family", status);
      FAIL("cannot query provisioned queues");
    }
    if ((info.roles & roles) == roles && info.provisioned_queue_count) {
      return hrx().device_get_queue(device, i, 0);
    }
  }
  FAIL("no provisioned queue supports the required operations");
  return nullptr;
}

class HrxTestFixture {
 protected:
  // Borrowed device owned by the CTS process until group release.
  hrx_device_t device_ = g_test_device;
  // Explicit execution queue used for uploads, readbacks, and transfer tests.
  hrx_queue_t transfer_queue_ =
      FindProvisionedQueue(device_, HRX_QUEUE_ROLE_TRANSFER);

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
