// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/external_registry.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "runtime/build_tools/bazel/test/hal_driver_registration_fixture.h"

namespace {

class ExternalDriverRegistryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(
        iree_hal_driver_registry_allocate(iree_allocator_system(), &registry_));
  }

  void TearDown() override { iree_hal_driver_registry_free(registry_); }

  iree_hal_driver_registry_t* registry_ = nullptr;
};

TEST_F(ExternalDriverRegistryTest, RegistersModulesInDeclaredOrder) {
  iree_hal_driver_registration_fixture_reset(0);
  IREE_EXPECT_OK(iree_hal_register_external_drivers(registry_));

  ASSERT_EQ(iree_hal_driver_registration_fixture_count(), 3u);
  EXPECT_EQ(iree_hal_driver_registration_fixture_at(0), 1);
  EXPECT_EQ(iree_hal_driver_registration_fixture_at(1), 2);
  EXPECT_EQ(iree_hal_driver_registration_fixture_at(2), 3);
}

TEST_F(ExternalDriverRegistryTest, StopsAtFirstRegistrationFailure) {
  iree_hal_driver_registration_fixture_reset(2);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED,
                        iree_hal_register_external_drivers(registry_));

  ASSERT_EQ(iree_hal_driver_registration_fixture_count(), 2u);
  EXPECT_EQ(iree_hal_driver_registration_fixture_at(0), 1);
  EXPECT_EQ(iree_hal_driver_registration_fixture_at(1), 2);
}

}  // namespace
