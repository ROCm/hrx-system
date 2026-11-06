// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/platform/linux/file.h"

#include "gtest/gtest.h"

namespace {

TEST(LinuxFileTest, ErrorAlwaysRepresentsFailure) {
  EXPECT_EQ(amdf_linux_error(EACCES),
            amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, EACCES));
  EXPECT_EQ(amdf_linux_error(0),
            amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, EIO));
}

}  // namespace
