// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/instance.h"

#include "gtest/gtest.h"

namespace {

TEST(InstanceLifetimeTest, DefaultsToProcessLifetime) {
  amdf_instance_create_info_t create_info = {};
  create_info.type = AMDF_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  create_info.structure_size = sizeof(create_info);
  amdf_instance_t* instance = nullptr;
  ASSERT_EQ(amdf_instance_create(&create_info, &instance), AMDF_STATUS_OK);
  EXPECT_EQ(instance->gpu, nullptr);
  EXPECT_EQ(amdf_instance_native_lifetime(instance),
            AMDF_NATIVE_LIFETIME_PROCESS);
  EXPECT_EQ(amdf_instance_destroy(instance), AMDF_STATUS_OK);
}

TEST(InstanceLifetimeTest, CopiesPolicyBeforeAnyEndpointIsOpened) {
  amdf_instance_create_info_t create_info = {};
  create_info.type = AMDF_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  create_info.structure_size = sizeof(create_info);
  create_info.native_lifetime = AMDF_NATIVE_LIFETIME_INSTANCE;
  amdf_instance_t* instance = nullptr;
  ASSERT_EQ(amdf_instance_create(&create_info, &instance), AMDF_STATUS_OK);
  create_info.native_lifetime = AMDF_NATIVE_LIFETIME_PROCESS;
  EXPECT_EQ(instance->gpu, nullptr);
  EXPECT_EQ(amdf_instance_native_lifetime(instance),
            AMDF_NATIVE_LIFETIME_INSTANCE);
  EXPECT_EQ(amdf_instance_destroy(instance), AMDF_STATUS_OK);
}

}  // namespace
