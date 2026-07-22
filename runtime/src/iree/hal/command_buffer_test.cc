// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/command_buffer.h"

#include <cstring>

#include "iree/testing/gtest.h"

namespace {

TEST(DispatchConfigCloneTest, ClearsAbsentRuntimeParameterStorage) {
  iree_hal_dispatch_runtime_parameter_list_t runtime_parameter_storage;
  memset(&runtime_parameter_storage, 0xA5, sizeof(runtime_parameter_storage));

  iree_hal_dispatch_config_t source = {};
  iree_hal_dispatch_config_t clone = {};
  iree_hal_dispatch_config_clone(source, &runtime_parameter_storage, &clone);

  EXPECT_EQ(nullptr, clone.runtime_parameters);
  EXPECT_EQ(0u, runtime_parameter_storage.count);
}

TEST(DispatchConfigCloneTest, ClonesPresentRuntimeParameters) {
  iree_hal_dispatch_runtime_parameter_list_t source_parameters = {};
  source_parameters.count = 1;
  source_parameters.patches[0].offset = 24;
  source_parameters.patches[0].length = sizeof(uint64_t);

  iree_hal_dispatch_config_t source = {};
  source.runtime_parameters = &source_parameters;
  iree_hal_dispatch_runtime_parameter_list_t runtime_parameter_storage = {};
  iree_hal_dispatch_config_t clone = {};
  iree_hal_dispatch_config_clone(source, &runtime_parameter_storage, &clone);

  EXPECT_EQ(&runtime_parameter_storage, clone.runtime_parameters);
  EXPECT_EQ(1u, runtime_parameter_storage.count);
  EXPECT_EQ(24u, runtime_parameter_storage.patches[0].offset);
  EXPECT_EQ(sizeof(uint64_t), runtime_parameter_storage.patches[0].length);
}

}  // namespace
