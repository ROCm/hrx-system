// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <errno.h>

#include "iree/hal/drivers/amdxdna/native_linux_kmq_internal.h"
#include "iree/testing/gtest.h"

namespace {

TEST(NativeLinuxKmqTest, BoAllocationHeapExhaustionIsRecoverable) {
  EXPECT_EQ(iree_hal_amdxdna_native_linux_bo_allocation_status_code(EAGAIN),
            IREE_STATUS_UNAVAILABLE);
  EXPECT_EQ(iree_hal_amdxdna_native_linux_bo_allocation_status_code(-EAGAIN),
            IREE_STATUS_UNAVAILABLE);
  EXPECT_EQ(iree_hal_amdxdna_native_linux_bo_allocation_status_code(ENOSPC),
            IREE_STATUS_UNAVAILABLE);
  EXPECT_EQ(iree_hal_amdxdna_native_linux_bo_allocation_status_code(-ENOSPC),
            IREE_STATUS_UNAVAILABLE);
}

TEST(NativeLinuxKmqTest, BoAllocationOtherErrorsPreserveStatus) {
  EXPECT_EQ(iree_hal_amdxdna_native_linux_bo_allocation_status_code(ENOMEM),
            IREE_STATUS_RESOURCE_EXHAUSTED);
  EXPECT_EQ(iree_hal_amdxdna_native_linux_bo_allocation_status_code(EINVAL),
            IREE_STATUS_INVALID_ARGUMENT);
}

}  // namespace
