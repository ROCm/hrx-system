// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/alignment.h"

#include "iree/testing/gtest.h"

namespace {

TEST(AlignmentTest, StoreLittleEndianF64) {
  double value = 0.0;
  iree_unaligned_store_le_f64(&value, 2.5);
  EXPECT_EQ(2.5, value);
}

}  // namespace
