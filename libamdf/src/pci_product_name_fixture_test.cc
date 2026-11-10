// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "gtest/gtest.h"
#include "libamdf/src/pci_product_name.h"

namespace {

TEST(PciProductNameFixtureTest, FirstBoardMatchOverridesOnlyItsParent) {
  amdf_pci_info_t pci = {
      .vendor_id = 0x1002,
      .device_id = 0x1586,
      .subsystem_vendor_id = 0x1028,
      .subsystem_device_id = 2,
      .revision_id = 0xC1,
  };
  EXPECT_STREQ(amdf_pci_product_name(&pci), "Board \"two\" \xCE\xBC");
  pci.revision_id = 0xC2;
  EXPECT_STREQ(amdf_pci_product_name(&pci), "Other board");
  pci.revision_id = 0xC1;
  pci.subsystem_device_id = 3;
  EXPECT_STREQ(amdf_pci_product_name(&pci), "Shared GPU name");
  pci.subsystem_vendor_id = 0x1029;
  pci.subsystem_device_id = 2;
  EXPECT_STREQ(amdf_pci_product_name(&pci), "Shared GPU name");
}

TEST(PciProductNameFixtureTest, SharedNamesUseOneStringAndLookupHasNoCursor) {
  amdf_pci_info_t pci = {
      .vendor_id = 0x1002,
      .device_id = 0x1586,
      .revision_id = 0xC1,
  };
  const char* base_name = amdf_pci_product_name(&pci);
  ASSERT_STREQ(base_name, "Shared GPU name");
  pci.subsystem_vendor_id = 0x1028;
  pci.subsystem_device_id = 1;
  EXPECT_EQ(amdf_pci_product_name(&pci), base_name);
  pci.device_id = 0xFFFF;
  pci.revision_id = 0xFF;
  EXPECT_STREQ(amdf_pci_product_name(&pci), "Last device");
  pci.device_id = 0;
  pci.revision_id = 0;
  EXPECT_STREQ(amdf_pci_product_name(&pci), "First device");
  pci.device_id = 1;
  EXPECT_EQ(amdf_pci_product_name(&pci), nullptr);
  EXPECT_STREQ(base_name, "Shared GPU name");
}

}  // namespace
