// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/pci.h"

#define AMDF_PCI_VENDOR_ID_ATI 0x1002u
#define AMDF_PCI_VENDOR_ID_AMD 0x1022u

bool amdf_pci_is_amd(const amdf_pci_info_t* pci) {
  return pci->vendor_id == AMDF_PCI_VENDOR_ID_ATI ||
         pci->vendor_id == AMDF_PCI_VENDOR_ID_AMD;
}

bool amdf_pci_parse_address(const char* text,
                            amdf_endpoint_pci_address_t* out_address) {
  const uint32_t widths[] = {4, 2, 2, 1};
  const char separators[] = {':', ':', '.', '\0'};
  uint32_t values[4] = {0};
  for (uint32_t field = 0; field < 4; ++field) {
    for (uint32_t i = 0; i < widths[field]; ++i) {
      const unsigned char character = (unsigned char)*text;
      const uint32_t digit =
          character >= '0' && character <= '9'   ? character - '0'
          : character >= 'a' && character <= 'f' ? character - 'a' + 10
          : character >= 'A' && character <= 'F' ? character - 'A' + 10
                                                 : UINT32_MAX;
      if (digit > 15) {
        return false;
      }
      values[field] = (values[field] << 4) | digit;
      ++text;
    }
    if (*text != separators[field]) {
      return false;
    }
    if (field != 3) {
      ++text;
    }
  }
  if (values[2] > 31 || values[3] > 7) {
    return false;
  }
  out_address->available = 1;
  out_address->domain = values[0];
  out_address->bus = values[1];
  out_address->device = values[2];
  out_address->function = values[3];
  return true;
}
