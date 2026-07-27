// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <cstring>

#include "hrx_test_fixture.hpp"

TEST_CASE_METHOD(HrxTestFixture, "Device has a name", "[device]") {
  char name[128] = {0};
  REQUIRE_OK(hrx().device_get_property(device_, HRX_DEVICE_PROPERTY_NAME, name,
                                       sizeof(name)));
  REQUIRE(strlen(name) > 0);
}

TEST_CASE_METHOD(HrxTestFixture, "Device has an architecture", "[device]") {
  char arch[64] = {0};
  REQUIRE_OK(hrx().device_get_property(
      device_, HRX_DEVICE_PROPERTY_ARCHITECTURE, arch, sizeof(arch)));
  REQUIRE(strlen(arch) > 0);
}

TEST_CASE_METHOD(HrxTestFixture,
                 "GPU device architecture is a compiler target ISA",
                 "[device][gpu]") {
  if (!is_gpu()) {
    SKIP("GPU architecture query is only validated on GPU devices");
  }

  char arch[64] = {0};
  REQUIRE_OK(hrx().device_get_property(
      device_, HRX_DEVICE_PROPERTY_ARCHITECTURE, arch, sizeof(arch)));
  REQUIRE(strncmp(arch, "gfx", 3) == 0);
}

TEST_CASE_METHOD(HrxTestFixture, "GPU device reports total memory",
                 "[device][gpu]") {
  if (!is_gpu()) {
    SKIP("GPU total memory is only validated on GPU devices");
  }

  uint64_t total_memory = 0;
  REQUIRE_OK(hrx().device_get_property(device_,
                                       HRX_DEVICE_PROPERTY_TOTAL_MEMORY,
                                       &total_memory, sizeof(total_memory)));
  REQUIRE(total_memory > 0);
}

TEST_CASE_METHOD(HrxTestFixture, "Device get_type matches accelerator",
                 "[device]") {
  hrx_accelerator_type_t type;
  REQUIRE_OK(hrx().device_get_type(device_, &type));
  REQUIRE(type == g_test_device_type);
}

TEST_CASE_METHOD(HrxTestFixture, "Device synchronize succeeds", "[device]") {
  REQUIRE_OK(hrx().device_synchronize(device_));
}

TEST_CASE_METHOD(HrxTestFixture, "Invalid device property returns error",
                 "[device]") {
  uint32_t val;
  hrx_status_t status = hrx().device_get_property(
      device_, (hrx_device_property_t)999, &val, sizeof(val));
  REQUIRE(!hrx_status_is_ok(status));
  hrx().status_ignore(status);
}
