// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdxdna/event.h"

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(EventTest, CreateAndRelease) {
  iree_hal_event_t* event = nullptr;
  IREE_ASSERT_OK(iree_hal_amdxdna_event_create(
      IREE_HAL_QUEUE_AFFINITY_ANY, IREE_HAL_EVENT_FLAG_NONE,
      iree_allocator_system(), &event));
  ASSERT_NE(event, nullptr);
  iree_hal_event_release(event);
}

}  // namespace
