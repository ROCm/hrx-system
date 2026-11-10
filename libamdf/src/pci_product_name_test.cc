// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/pci_product_name.h"

#include <cstring>

#include "gtest/gtest.h"

namespace {

TEST(PciProductNameTest, ProductIdentityIncludesTheRevision) {
  amdf_pci_info_t pci = {.vendor_id = 0x1002, .device_id = 0x1586};
  for (uint32_t revision = 0; revision < 256; ++revision) {
    SCOPED_TRACE(revision);
    pci.revision_id = revision;
    const char* expected = nullptr;
    switch (revision) {
      case 0xC1:
      case 0xC3:
      case 0xC6:
      case 0xD1:
        expected = "AMD Radeon 8060S Graphics";
        break;
      case 0xC2:
      case 0xC4:
      case 0xD2:
      case 0xD4:
        expected = "AMD Radeon 8050S Graphics";
        break;
      case 0xD5:
        expected = "AMD Radeon 8040S Graphics";
        break;
    }
    EXPECT_STREQ(amdf_pci_product_name(&pci), expected);
  }
}

TEST(PciProductNameTest, NamesDiscreteIntegratedAndComputeProducts) {
  struct Case {
    // PCI device identifier.
    uint32_t device;
    // PCI revision identifier.
    uint32_t revision;
    // Public registry spelling for this product.
    const char* name;
  };
  for (const auto& value : {Case{0x1114, 0xC2, "AMD Radeon 860M Graphics"},
                            Case{0x744C, 0xC8, "AMD Radeon RX 7900 XTX"},
                            Case{0x744C, 0xCC, "AMD Radeon RX 7900 XT"},
                            Case{0x74A1, 0, "AMD Instinct MI300X"},
                            Case{0x75A8, 0, "AMD Instinct MI350P"}}) {
    amdf_pci_info_t pci = {
        .vendor_id = 0x1002,
        .device_id = value.device,
        .revision_id = value.revision,
    };
    const char* name = amdf_pci_product_name(&pci);
    ASSERT_NE(name, nullptr);
    EXPECT_STREQ(name, value.name);
    EXPECT_LT(std::strlen(name), AMDF_ENDPOINT_NAME_CAPACITY);
    EXPECT_EQ(amdf_pci_product_name(&pci), name);
  }
}

TEST(PciProductNameTest, UnknownIdentitiesRetainNoInventedProduct) {
  amdf_pci_info_t pci = {
      .vendor_id = 0x1002,
      .device_id = 0xFFFF,
      .revision_id = 0xFF,
  };
  EXPECT_EQ(amdf_pci_product_name(&pci), nullptr);
  pci.device_id = 0x1586;
  pci.revision_id = 0xC1;
  for (uint32_t vendor : {0u, 0x1022u, 0x8086u, 0x10DEu}) {
    pci.vendor_id = vendor;
    EXPECT_EQ(amdf_pci_product_name(&pci), nullptr);
  }
}

}  // namespace
