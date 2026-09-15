// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/pci.h"

#include "gtest/gtest.h"

namespace {

TEST(PciTest, RecognizesAmdEndpointVendors) {
  amdf_pci_info_t pci = {};
  pci.vendor_id = 0x1002u;
  EXPECT_TRUE(amdf_pci_is_amd(&pci));
  pci.vendor_id = 0x1022u;
  EXPECT_TRUE(amdf_pci_is_amd(&pci));
  pci.vendor_id = 0x8086u;
  EXPECT_FALSE(amdf_pci_is_amd(&pci));
}

}  // namespace
