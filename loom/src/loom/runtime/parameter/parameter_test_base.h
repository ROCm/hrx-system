// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_RUNTIME_PARAMETER_PARAMETER_TEST_BASE_H_
#define LOOM_RUNTIME_PARAMETER_PARAMETER_TEST_BASE_H_

#include "iree/testing/gtest.h"
#include "loom/runtime/device/device.h"

namespace loom_runtime::testing {

// Gather and module tests share one native application per executor size.
// Individual cases own their files, buffers, timelines, and retirement joins.
class ParameterTest : public ::testing::TestWithParam<uint32_t> {
 protected:
  void SetUp() override;
  void TearDown() override;

  // Host allocation policy, alive through final queue retirement.
  iree_allocator_t allocator_ = iree_allocator_system();
  // Borrowed device with the requested worker-count configuration.
  loom_runtime_device_t* owner_ = nullptr;
  // Borrowed fixed backing shared after each case returns its reservations.
  loom_runtime_pool_t* storage_owner_ = nullptr;
  // Borrowed physical device, retained through global environment teardown.
  iree_hal_device_t* device_ = nullptr;
  // Borrowed exact transfer queue used by all tested operations.
  iree_hal_queue_t* queue_ = nullptr;
};

}  // namespace loom_runtime::testing

#endif  // LOOM_RUNTIME_PARAMETER_PARAMETER_TEST_BASE_H_
