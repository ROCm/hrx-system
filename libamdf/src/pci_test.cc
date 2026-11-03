// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/pci.h"

#include <cstring>

#include "gtest/gtest.h"

namespace {

TEST(PciTest, RecognizesAmdEndpointVendors) {
  amdf_pci_info_t pci = {
      .vendor_id = 0x1002u,
  };
  EXPECT_TRUE(amdf_pci_is_amd(&pci));
  pci.vendor_id = 0x1022u;
  EXPECT_TRUE(amdf_pci_is_amd(&pci));
  pci.vendor_id = 0x8086u;
  EXPECT_FALSE(amdf_pci_is_amd(&pci));
}

TEST(PciTest, ParsesCanonicalAddressesWithoutChangingOutputHeader) {
  struct Case {
    // Canonical DBDF from the native PCI namespace.
    const char* text;
    // Expected PCI domain.
    uint32_t domain;
    // Expected PCI bus.
    uint32_t bus;
    // Expected PCI device.
    uint32_t device;
    // Expected PCI function.
    uint32_t function;
  };
  for (const auto& value : {Case{"0000:00:00.0", 0, 0, 0, 0},
                            Case{"0012:ab:1d.6", 0x12, 0xab, 0x1d, 6},
                            Case{"FFFF:FF:1F.7", 0xffff, 0xff, 0x1f, 7}}) {
    SCOPED_TRACE(value.text);
    amdf_endpoint_pci_address_t address = {
        .type = AMDF_STRUCTURE_TYPE_ENDPOINT_PCI_ADDRESS,
        .structure_size = sizeof(address),
    };
    ASSERT_TRUE(amdf_pci_parse_address(value.text, &address));
    EXPECT_EQ(address.type, AMDF_STRUCTURE_TYPE_ENDPOINT_PCI_ADDRESS);
    EXPECT_EQ(address.structure_size, sizeof(address));
    EXPECT_EQ(address.next, nullptr);
    EXPECT_EQ(address.available, 1u);
    EXPECT_EQ(address.domain, value.domain);
    EXPECT_EQ(address.bus, value.bus);
    EXPECT_EQ(address.device, value.device);
    EXPECT_EQ(address.function, value.function);
  }
}

TEST(PciTest, InvalidAddressesLeaveOutputUnchanged) {
  for (const char* text :
       {"", "0000", "0000:00:00", "0000:00:00.", "0000:00:00.00",
        "0000:00:20.0", "0000:00:00.8", "00000:00:00.0", "0000:gg:00.0",
        "0000:00:00.0 ", " 000:00:00.0", "0000:00:00.0/", "0000-00:00.0"}) {
    SCOPED_TRACE(text);
    amdf_endpoint_pci_address_t address;
    std::memset(&address, 0xa5, sizeof(address));
    const auto original = address;
    EXPECT_FALSE(amdf_pci_parse_address(text, &address));
    EXPECT_EQ(std::memcmp(&address, &original, sizeof(address)), 0);
  }
}

}  // namespace
