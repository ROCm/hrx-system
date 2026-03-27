// Copyright 2026 The Pyre Authors
// SPDX-License-Identifier: Apache-2.0

#include "pyre_test_fixture.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cstring>

TEST_CASE_METHOD(PyreTestFixture, "Device has a name", "[device]") {
  char name[128] = {0};
  REQUIRE_OK(pyre().device_get_property(
      device_, PYRE_DEVICE_PROPERTY_NAME, name, sizeof(name)));
  REQUIRE(strlen(name) > 0);
}

TEST_CASE_METHOD(PyreTestFixture, "Device has an architecture", "[device]") {
  char arch[64] = {0};
  REQUIRE_OK(pyre().device_get_property(
      device_, PYRE_DEVICE_PROPERTY_ARCHITECTURE, arch, sizeof(arch)));
  REQUIRE(strlen(arch) > 0);
}

TEST_CASE_METHOD(PyreTestFixture, "Device get_type matches accelerator",
                 "[device]") {
  pyre_accelerator_type_t type;
  REQUIRE_OK(pyre().device_get_type(device_, &type));
  REQUIRE(type == g_test_device_type);
}

TEST_CASE_METHOD(PyreTestFixture, "Device synchronize succeeds", "[device]") {
  REQUIRE_OK(pyre().device_synchronize(device_));
}

TEST_CASE_METHOD(PyreTestFixture, "Invalid device property returns error",
                 "[device]") {
  uint32_t val;
  pyre_status_t status = pyre().device_get_property(
      device_, (pyre_device_property_t)999, &val, sizeof(val));
  REQUIRE(!pyre_status_is_ok(status));
  pyre().status_ignore(status);
}
