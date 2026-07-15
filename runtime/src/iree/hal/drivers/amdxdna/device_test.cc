// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdxdna/api.h"
#include "iree/testing/gtest.h"

namespace {

TEST(DeviceTest, OptionsInitializeClearsAllFields) {
  iree_hal_amdxdna_device_params params = {};
  params.n_core_rows = 4;
  params.n_core_cols = 5;
  params.device_path = IREE_SV("device");
  params.power_mode = IREE_SV("turbo");

  iree_hal_amdxdna_device_options_initialize(&params);

  EXPECT_EQ(params.n_core_rows, 0);
  EXPECT_EQ(params.n_core_cols, 0);
  EXPECT_TRUE(iree_string_view_is_empty(params.device_path));
  EXPECT_TRUE(iree_string_view_is_empty(params.power_mode));
}

}  // namespace
