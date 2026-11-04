// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/dxg/rocdxg/native_allocation_flags.h"

#include "gtest/gtest.h"

namespace {

TEST(NativeAllocationFlagsTest, PreservesIndependentNativePolicies) {
  EXPECT_EQ(amdf::rocdxg_bridge::EncodeNativeAllocationFlags(0), 0u);
  EXPECT_EQ(amdf::rocdxg_bridge::EncodeNativeAllocationFlags(
                AMDF_DXG_BRIDGE_GPU_ALLOCATION_FLAG_FINE_GRAIN),
            thunk_proxy::kFineGrain);
  EXPECT_EQ(amdf::rocdxg_bridge::EncodeNativeAllocationFlags(
                AMDF_DXG_BRIDGE_GPU_ALLOCATION_FLAG_DEVICE_UNCACHED),
            thunk_proxy::kKernarg);
  EXPECT_EQ(amdf::rocdxg_bridge::EncodeNativeAllocationFlags(
                AMDF_DXG_BRIDGE_GPU_ALLOCATION_FLAG_FINE_GRAIN |
                AMDF_DXG_BRIDGE_GPU_ALLOCATION_FLAG_QUEUE_STORAGE |
                AMDF_DXG_BRIDGE_GPU_ALLOCATION_FLAG_DEVICE_UNCACHED),
            thunk_proxy::kFineGrain | thunk_proxy::kKernarg);
}

}  // namespace
